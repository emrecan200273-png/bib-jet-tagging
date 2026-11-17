"""Two attention blocks, optionally using distances from BIB timing profiles.

Cells enter in the training-standardized order t, log-relative energy, R, z,
delta-eta, delta-phi. True mask entries mark real cells.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from analysis.models.cell_models import attention_key_mask, jet_classifier


WIDTH = 32
HEADS = 4
HEAD_WIDTH = WIDTH // HEADS
GEOMETRY_DIM = 7
DROP = 0.1
C_M_PER_NS = 0.299792458


def _augment_qkv(query, key, value, coordinates, weights):
    """Add -weight * (position_i - position_j)^2 to the attention scores."""
    # The position_i^2 term is constant for one query, so softmax cancels it.
    geometry = coordinates.to(query.dtype).unsqueeze(1).expand(-1, query.shape[1], -1, -1)
    weight = weights.to(query.dtype)[None, :, None, :]
    content_width = query.shape[-1]
    query = torch.cat((
        query / math.sqrt(content_width),
        2 * weight * geometry,
        -weight.expand_as(geometry),
    ), dim=-1)
    key = torch.cat((key, geometry, geometry.square()), dim=-1)
    # A common head width, aligned to eight, keeps optimized SDPA available.
    padding = (-query.shape[-1]) % 8
    return (
        F.pad(query, (0, padding)),
        F.pad(key, (0, padding)),
        F.pad(value, (0, query.shape[-1] + padding - content_width)),
    )


class _ContextBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        # Softplus initializes each distance weight at 0.1 and keeps it positive.
        self.log_distance_weights = nn.Parameter(torch.full(
            (HEADS, GEOMETRY_DIM), math.log(math.expm1(0.1)),
        ))
        self.attention_output = nn.Linear(WIDTH, WIDTH)
        self.feedforward_norm = nn.LayerNorm(WIDTH)
        self.feedforward = nn.Sequential(
            nn.Linear(WIDTH, 64), nn.GELU(), nn.Dropout(DROP), nn.Linear(64, WIDTH),
        )
        self.dropout = nn.Dropout(DROP)

    def forward(self, cells, coordinates, valid_keys):
        batch, count, _ = cells.shape
        qkv = self.qkv(self.attention_norm(cells)).reshape(batch, count, 3, HEADS, HEAD_WIDTH)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query, key, value = _augment_qkv(
            query, key, value, coordinates, F.softplus(self.log_distance_weights),
        )
        # Output dropout avoids allocating a dropout mask for every cell pair.
        contextual = F.scaled_dot_product_attention(
            query, key, value, attn_mask=valid_keys[:, None, None, :], scale=1.0,
        )[..., :HEAD_WIDTH]
        contextual = contextual.transpose(1, 2).reshape(batch, count, WIDTH)
        cells = cells + self.dropout(self.attention_output(contextual))
        return cells + self.dropout(self.feedforward(self.feedforward_norm(cells)))


class DeepCellTransformer(nn.Module):
    """Compare cells twice, then pool them with one learned query."""

    def __init__(self, jet_dim, cell_dim, *, physics=False, cell_center, cell_scale):
        super().__init__()
        if cell_dim != 6:
            raise ValueError("Deep cell attention expects the six standard cell features")
        center = torch.as_tensor(cell_center, dtype=torch.float32)
        scale = torch.as_tensor(cell_scale, dtype=torch.float32)
        if (
            center.shape != (cell_dim,) or scale.shape != (cell_dim,)
            or not torch.isfinite(center).all() or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("Expected finite training centers and positive training scales")
        self.register_buffer("cell_center", center.detach().clone())
        self.register_buffer("cell_scale", scale.detach().clone())
        self.physics = physics
        self.cell_projection = nn.Linear(cell_dim, WIDTH)
        self.blocks = nn.ModuleList((_ContextBlock(), _ContextBlock()))
        self.class_query = nn.Parameter(torch.empty(1, 1, WIDTH))
        nn.init.normal_(self.class_query, std=0.02)
        self.query_norm = nn.LayerNorm(WIDTH)
        self.query_projection = nn.Linear(WIDTH, WIDTH)
        self.readout_norm = nn.LayerNorm(WIDTH)
        self.readout_kv = nn.Linear(WIDTH, 2 * WIDTH)
        self.readout_output = nn.Linear(WIDTH, WIDTH)
        self.classifier = jet_classifier(jet_dim + WIDTH, DROP)

    def _geometry(self, cells, valid):
        base = cells[..., (0, 2, 3, 4, 5)]
        if self.physics:
            raw = cells[..., :4] * self.cell_scale[:4] + self.cell_center[:4]
            time, _, radius, z = raw.unbind(dim=-1)
            distance = torch.hypot(radius, z)
            profiles = torch.stack((z - distance, -z - distance), dim=-1) / C_M_PER_NS
            extra = (time.unsqueeze(-1) - profiles) / self.cell_scale[0]
            # Removing a per-jet constant leaves all pair distances unchanged
            # and avoids cancellation from a large common timing offset.
            extra = extra.masked_fill(~valid.unsqueeze(-1), 0)
            count = valid.sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1)
            extra = extra - extra.sum(dim=1, keepdim=True) / count
        else:
            extra = cells.new_zeros((*cells.shape[:2], 2))
        return torch.cat((base, extra), dim=-1).masked_fill(~valid.unsqueeze(-1), 0)

    def _readout(self, cells, valid_keys):
        batch, count, _ = cells.shape
        query = self.query_projection(self.query_norm(self.class_query))
        query = query.reshape(1, 1, HEADS, HEAD_WIDTH).transpose(1, 2).expand(batch, -1, -1, -1)
        kv = self.readout_kv(self.readout_norm(cells)).reshape(batch, count, 2, HEADS, HEAD_WIDTH)
        key, value = kv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        pooled = F.scaled_dot_product_attention(query, key, value, attn_mask=valid_keys[:, None, None, :])
        return self.readout_output(pooled.reshape(batch, WIDTH))

    def forward(self, jets, cells, mask):
        batch, count, _ = cells.shape
        if count == 0:
            pooled = jets.new_zeros((batch, WIDTH))
        else:
            valid = mask.bool()
            cells = cells.masked_fill(~valid.unsqueeze(-1), 0)
            valid_keys, empty = attention_key_mask(valid)
            coordinates = self._geometry(cells, valid)
            encoded = self.cell_projection(cells)
            for block in self.blocks:
                encoded = block(encoded, coordinates, valid_keys)
            pooled = self._readout(encoded, valid_keys).masked_fill(empty, 0)
        return self.classifier(torch.cat((jets, pooled), dim=-1)).squeeze(-1)
