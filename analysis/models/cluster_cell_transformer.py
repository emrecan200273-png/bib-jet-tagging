"""Pool cell information inside each topocluster, then attend across clusters."""

import torch
from torch import nn

from analysis.models.cell_models import jet_classifier


WIDTH = 32
HEADS = 4
DROPOUT = 0.1


class ClusterCellTransformer(nn.Module):
    def __init__(self, jet_dim, cell_dim):
        super().__init__()
        if cell_dim != 7:
            raise ValueError("Expected six cell features followed by the original cluster ID")
        self.cell_encoder = nn.Sequential(nn.Linear(cell_dim - 1, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))
        self.cell_attention = nn.Linear(WIDTH, HEADS)
        self.summary = nn.Parameter(torch.empty(1, 1, WIDTH))
        nn.init.normal_(self.summary, std=0.02)
        self.cluster_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(WIDTH, HEADS, 2 * WIDTH, DROPOUT, activation="gelu", batch_first=True)
            for _ in range(2)
        ])
        self.classifier = jet_classifier(jet_dim + WIDTH)

    def _pool_clusters(self, cells, valid):
        # The stored ID groups cells for pooling. The encoder reads the six cell features.
        cluster_ids = cells[..., -1].masked_fill(~valid, -1).long()
        cluster_count = int(cluster_ids.amax().item()) + 1 if cluster_ids.numel() else 0
        if not cluster_count:
            return cells.new_zeros((len(cells), 0, WIDTH)), valid[:, :0]

        cell_embeddings = self.cell_encoder(cells[..., :-1].masked_fill(~valid[..., None], 0))
        weight_logits = self.cell_attention(cell_embeddings)
        cluster_indices = cluster_ids.clamp_min(0)[..., None].expand(-1, -1, HEADS)
        # Subtract the largest logit in each cluster before exponentiating.
        with torch.no_grad():
            cluster_maxima = weight_logits.new_full((len(cells), cluster_count, HEADS), -torch.inf)
            cluster_maxima.scatter_reduce_(
                1, cluster_indices, weight_logits.masked_fill(~valid[..., None], -torch.inf), reduce="amax",
            )
            cell_maxima = cluster_maxima.gather(1, cluster_indices).masked_fill(~valid[..., None], 0)
        weights = (weight_logits - cell_maxima).masked_fill(~valid[..., None], -torch.inf).exp()
        weight_sums = weights.new_zeros((len(cells), cluster_count, HEADS)).scatter_add(1, cluster_indices, weights)
        weighted_cells = cell_embeddings.unflatten(-1, (HEADS, WIDTH // HEADS)) * weights[..., None]
        cluster_sums = weighted_cells.new_zeros((len(cells), cluster_count, HEADS, WIDTH // HEADS)).scatter_add(
            1, cluster_indices[..., None].expand_as(weighted_cells), weighted_cells,
        )
        cluster_embeddings = (cluster_sums / weight_sums.clamp_min(1e-12)[..., None]).flatten(2)
        return cluster_embeddings, weight_sums[..., 0] > 0

    def forward(self, jets, cells, mask):
        cluster_embeddings, valid_clusters = self._pool_clusters(cells, mask.bool())
        # The first token summarizes the jet and is valid even without clusters.
        tokens = torch.cat((self.summary.expand(len(jets), -1, -1), cluster_embeddings), dim=1)
        padding_mask = torch.cat((valid_clusters.new_zeros((len(jets), 1)), ~valid_clusters), dim=1)
        for layer in self.cluster_layers:
            tokens = layer(tokens, src_key_padding_mask=padding_mask)
        jet_summary = tokens[:, 0]
        return self.classifier(torch.cat((jets, jet_summary), dim=-1)).squeeze(-1)
