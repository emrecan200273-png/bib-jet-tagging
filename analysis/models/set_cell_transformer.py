"""Set Transformer with two induced attention blocks and one pooling seed.

Every valid cell participates, with no order or padding position as input.
The inducing vectors are learned summaries, not selected detector cells.
"""

import torch
from torch import nn
from torch.nn import functional as F

from analysis.models.cell_models import attention_key_mask, jet_classifier


WIDTH = 32
HEADS = 4
HEAD_WIDTH = WIDTH // HEADS
INDUCING_POINTS = 8
DROPOUT = 0.1


class _AttentionBlock(nn.Module):
    """Cross-attention with residual, normalization and a per-token MLP."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(WIDTH, WIDTH)
        self.key_value = nn.Linear(WIDTH, 2 * WIDTH)
        self.output = nn.Linear(WIDTH, WIDTH)
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.feedforward = nn.Sequential(
            nn.Linear(WIDTH, 64), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(64, WIDTH),
        )
        self.feedforward_norm = nn.LayerNorm(WIDTH)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, queries, context, valid_keys=None):
        batch, n_queries, _ = queries.shape
        n_keys = context.shape[1]
        query = self.query(queries).reshape(batch, n_queries, HEADS, HEAD_WIDTH).transpose(1, 2)
        key_value = self.key_value(context).reshape(batch, n_keys, 2, HEADS, HEAD_WIDTH)
        key, value = key_value.permute(2, 0, 3, 1, 4).unbind(0)
        mask = valid_keys[:, None, None, :] if valid_keys is not None else None
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, n_queries, WIDTH)
        hidden = self.attention_norm(queries + self.dropout(self.output(attended)))
        return self.feedforward_norm(hidden + self.dropout(self.feedforward(hidden)))


class _InducedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, INDUCING_POINTS, WIDTH))
        nn.init.normal_(self.inducing, std=0.02)
        self.summarize = _AttentionBlock()
        self.update = _AttentionBlock()

    def forward(self, cells, valid_keys):
        summaries = self.summarize(self.inducing.expand(cells.shape[0], -1, -1), cells, valid_keys)
        return self.update(cells, summaries)


class SetCellTransformer(nn.Module):
    """Encode cell interactions through eight inducing points per block."""

    def __init__(self, jet_dim, cell_dim):
        super().__init__()
        self.cell_projection = nn.Linear(cell_dim, WIDTH)
        self.encoder = nn.ModuleList((_InducedAttention(), _InducedAttention()))
        self.pool_seed = nn.Parameter(torch.empty(1, 1, WIDTH))
        nn.init.normal_(self.pool_seed, std=0.02)
        self.pool = _AttentionBlock()
        self.classifier = jet_classifier(jet_dim + WIDTH)

    def forward(self, jets, cells, mask):
        batch, count, _ = cells.shape
        if count == 0:
            pooled = jets.new_zeros((batch, WIDTH))
        else:
            valid = mask.bool()
            valid_keys, empty = attention_key_mask(valid)
            encoded = self.cell_projection(cells.masked_fill(~valid.unsqueeze(-1), 0))
            for block in self.encoder:
                encoded = block(encoded, valid_keys)
            pooled = self.pool(self.pool_seed.expand(batch, -1, -1), encoded, valid_keys)[:, 0]
            pooled = pooled.masked_fill(empty, 0)
        return self.classifier(torch.cat((jets, pooled), dim=-1)).squeeze(-1)
