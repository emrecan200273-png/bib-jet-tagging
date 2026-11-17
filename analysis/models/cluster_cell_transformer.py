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
        cluster = cells[..., -1].masked_fill(~valid, -1).long()
        count = int(cluster.amax().item()) + 1 if cluster.numel() else 0
        if not count:
            return cells.new_zeros((len(cells), 0, WIDTH)), valid[:, :0]

        encoded = self.cell_encoder(cells[..., :-1].masked_fill(~valid[..., None], 0))
        logits = self.cell_attention(encoded)
        cluster = cluster.clamp_min(0)[..., None].expand(-1, -1, HEADS)
        with torch.no_grad():
            maxima = logits.new_full((len(cells), count, HEADS), -torch.inf)
            maxima.scatter_reduce_(1, cluster, logits.masked_fill(~valid[..., None], -torch.inf), reduce="amax")
            shift = maxima.gather(1, cluster).masked_fill(~valid[..., None], 0)
        weights = (logits - shift).masked_fill(~valid[..., None], -torch.inf).exp()
        totals = weights.new_zeros((len(cells), count, HEADS)).scatter_add(1, cluster, weights)
        values = encoded.unflatten(-1, (HEADS, WIDTH // HEADS)) * weights[..., None]
        pooled = values.new_zeros((len(cells), count, HEADS, WIDTH // HEADS)).scatter_add(
            1, cluster[..., None].expand_as(values), values,
        )
        return (pooled / totals.clamp_min(1e-12)[..., None]).flatten(2), totals[..., 0] > 0

    def forward(self, jets, cells, mask):
        clusters, valid = self._pool_clusters(cells, mask.bool())
        tokens = torch.cat((self.summary.expand(len(jets), -1, -1), clusters), dim=1)
        padding = torch.cat((valid.new_zeros((len(jets), 1)), ~valid), dim=1)
        for layer in self.cluster_layers:
            tokens = layer(tokens, src_key_padding_mask=padding)
        return self.classifier(torch.cat((jets, tokens[:, 0]), dim=-1)).squeeze(-1)
