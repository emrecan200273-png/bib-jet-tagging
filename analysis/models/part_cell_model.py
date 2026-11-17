"""Small ParT-inspired attention model for calorimeter-cell sets.

Cells are standardized with training-split statistics before entering the
model. The mask is True for a real cell. No ordering or padding position is a
model feature.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from analysis.models.cell_models import attention_key_mask, jet_classifier


class PairwiseCellTransformer(nn.Module):
    """Add learned timing and geometry biases to cell-to-cell attention."""

    # Layout: time, log-relative energy, R, z, delta-eta, delta-phi.
    pair_features = (0, 3, 2, 4, 5)
    width = 32
    heads = 4
    pair_width = 8
    dropout = 0.1

    def __init__(self, jet_dim, cell_dim):
        super().__init__()
        if cell_dim != 6:
            raise ValueError("Pairwise cell attention expects the six standard cell features")
        self.cell_projection = nn.Linear(cell_dim, self.width)
        self.attention_norm = nn.LayerNorm(self.width)
        self.qkv = nn.Linear(self.width, 3 * self.width)
        self.attention_output = nn.Linear(self.width, self.width)
        self.pair_projection = nn.Linear(len(self.pair_features), self.pair_width, bias=False)
        self.pair_hidden_bias = nn.Parameter(torch.zeros(self.pair_width))
        self.pair_head_weights = nn.Parameter(torch.empty(self.heads, self.pair_width))
        nn.init.normal_(self.pair_head_weights, std=0.05)
        self.feedforward_norm = nn.LayerNorm(self.width)
        self.feedforward = nn.Sequential(
            nn.Linear(self.width, 64), nn.GELU(), nn.Dropout(self.dropout),
            nn.Linear(64, self.width),
        )
        self.residual_dropout = nn.Dropout(self.dropout)
        self.readout_norm = nn.LayerNorm(self.width)
        self.readout_score = nn.Linear(self.width, 1)
        self.classifier = jet_classifier(jet_dim + self.width, self.dropout)

    def _pair_bias(self, cells):
        """Evaluate an MLP on signed differences without a pair-feature grid."""
        batch, count, _ = cells.shape
        # Projection commutes with subtraction. A common scale is applied to
        # all standardized differences; these are not physical-unit distances.
        projected = self.pair_projection(cells[..., self.pair_features]) / math.sqrt(2.0)
        hidden_bias = self.pair_hidden_bias.to(projected.dtype)
        head_weights = self.pair_head_weights.to(projected.dtype)
        bias = projected.new_zeros((batch, self.heads, count, count))
        for index in range(self.pair_width):
            value = projected[..., index]
            hidden = torch.tanh(value.unsqueeze(2) - value.unsqueeze(1) + hidden_bias[index])
            bias = torch.addcmul(bias, hidden.unsqueeze(1), head_weights[:, index].view(1, -1, 1, 1))
        return bias

    def forward(self, jets, cells, mask):
        batch, count, _ = cells.shape
        if count == 0:
            pooled = jets.new_zeros((batch, self.width))
        else:
            valid = mask.bool()
            safe_cells = cells.masked_fill(~valid.unsqueeze(-1), 0)
            valid_keys, _ = attention_key_mask(valid)

            x = self.cell_projection(safe_cells)
            qkv = self.qkv(self.attention_norm(x)).reshape(batch, count, 3, self.heads, -1)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            bias = self._pair_bias(safe_cells)
            bias.masked_fill_(~valid_keys[:, None, None, :], -torch.inf)
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attended = attended.transpose(1, 2).reshape(batch, count, self.width)
            x = x + self.residual_dropout(self.attention_output(attended))
            x = x + self.residual_dropout(self.feedforward(self.feedforward_norm(x)))

            x = self.readout_norm(x)
            logits = self.readout_score(x).squeeze(-1).masked_fill(~valid_keys, -torch.inf)
            weights = logits.softmax(dim=1).masked_fill(~valid, 0)
            pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return self.classifier(torch.cat((jets, pooled), dim=-1)).squeeze(-1)
