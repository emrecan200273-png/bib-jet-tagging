"""Set Transformer with two induced attention blocks and one pooling seed.

Every valid cell participates, with no order or padding position as input.
Learned queries let each block summarize the full set in eight vectors.
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
        batch_size, query_count, _ = queries.shape
        key_count = context.shape[1]
        query = self.query(queries).reshape(batch_size, query_count, HEADS, HEAD_WIDTH).transpose(1, 2)
        key_value = self.key_value(context).reshape(batch_size, key_count, 2, HEADS, HEAD_WIDTH)
        key, value = key_value.permute(2, 0, 3, 1, 4).unbind(0)
        attention_mask = valid_keys[:, None, None, :] if valid_keys is not None else None
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        attended = attended.transpose(1, 2).reshape(batch_size, query_count, WIDTH)
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
        # Learned queries read all valid cells, then each cell reads the summaries.
        inducing_queries = self.inducing.expand(cells.shape[0], -1, -1)
        summaries = self.summarize(inducing_queries, cells, valid_keys)
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
        batch_size, cell_count, _ = cells.shape
        if cell_count == 0:
            jet_summary = jets.new_zeros((batch_size, WIDTH))
        else:
            valid = mask.bool()
            valid_keys, empty_jets = attention_key_mask(valid)
            cell_embeddings = self.cell_projection(cells.masked_fill(~valid.unsqueeze(-1), 0))
            for block in self.encoder:
                cell_embeddings = block(cell_embeddings, valid_keys)
            # One learned query pools the whole jet. Empty jets use a zero summary.
            pool_query = self.pool_seed.expand(batch_size, -1, -1)
            jet_summary = self.pool(pool_query, cell_embeddings, valid_keys)[:, 0]
            jet_summary = jet_summary.masked_fill(empty_jets, 0)
        return self.classifier(torch.cat((jets, jet_summary), dim=-1)).squeeze(-1)
