"""Small classifiers with shared jet inputs and optional calorimeter cells.

All models return one logit per jet. A True mask entry marks a real cell.
"""

import torch
from torch import nn


WIDTH = 32
DROPOUT = 0.1


def _classifier(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.SiLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(64, 32),
        nn.SiLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(32, 1),
    )


class JetMLP(nn.Module):
    def __init__(self, jet_dim):
        super().__init__()
        self.classifier = _classifier(jet_dim)

    def forward(self, jets, cells, mask):
        return self.classifier(jets).squeeze(-1)


class DeepSets(nn.Module):
    def __init__(self, jet_dim, cell_dim):
        super().__init__()
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.SiLU(),
        )
        self.classifier = _classifier(jet_dim + 2 * WIDTH)

    def forward(self, jets, cells, mask):
        valid = mask.bool().unsqueeze(-1)
        if cells.shape[1] == 0:
            pooled = cells.new_zeros((cells.shape[0], 2 * WIDTH))
        else:
            encoded = self.cell_encoder(cells.masked_fill(~valid, 0))
            count = valid.sum(dim=1)
            mean = encoded.masked_fill(~valid, 0).sum(dim=1) / count.clamp_min(1)
            maximum = encoded.masked_fill(~valid, -torch.inf).amax(dim=1)
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            pooled = torch.cat((mean, maximum), dim=-1)
        return self.classifier(torch.cat((jets, pooled), dim=-1)).squeeze(-1)


class CellTransformer(nn.Module):
    def __init__(self, jet_dim, cell_dim):
        super().__init__()
        self.cell_projection = nn.Linear(cell_dim, WIDTH)
        self.summary_token = nn.Parameter(torch.empty(1, 1, WIDTH))
        nn.init.normal_(self.summary_token, std=0.02)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=WIDTH, nhead=4, dim_feedforward=64, dropout=DROPOUT,
            activation="gelu", batch_first=True,
        )
        self.classifier = _classifier(jet_dim + WIDTH)

    def forward(self, jets, cells, mask):
        valid = mask.bool()
        encoded = self.cell_projection(cells.masked_fill(~valid.unsqueeze(-1), 0))
        summary = self.summary_token.to(encoded.dtype).expand(cells.shape[0], -1, -1)
        tokens = torch.cat((summary, encoded), dim=1)
        summary_valid = torch.ones((cells.shape[0], 1), dtype=torch.bool, device=valid.device)
        padding = ~torch.cat((summary_valid, valid), dim=1)
        pooled = self.encoder(tokens, src_key_padding_mask=padding)[:, 0]
        return self.classifier(torch.cat((jets, pooled), dim=-1)).squeeze(-1)


def build_model(name, jet_dim, cell_dim, **options):
    """Build a classifier consuming (jets, padded cells, valid-cell mask)."""
    if name == "jet_mlp":
        return JetMLP(jet_dim)
    if name == "deep_sets":
        return DeepSets(jet_dim, cell_dim)
    if name == "transformer":
        return CellTransformer(jet_dim, cell_dim)
    if name == "pairwise_transformer":
        from part_cell_model import PairwiseCellTransformer

        return PairwiseCellTransformer(jet_dim, cell_dim)
    if name == "particle_net_lite":
        from particle_net_cell_model import ParticleNetLiteCells

        return ParticleNetLiteCells(jet_dim, cell_dim, **options)
    if name in ("deep_cell_transformer", "bib_cell_transformer"):
        from deep_cell_transformer import DeepCellTransformer

        return DeepCellTransformer(jet_dim, cell_dim, physics=name == "bib_cell_transformer", **options)
    raise ValueError(f"Unknown model {name!r}")
