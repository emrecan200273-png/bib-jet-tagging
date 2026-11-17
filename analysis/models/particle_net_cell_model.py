"""ParticleNet-Lite adapted to calorimeter cells.

Each block links cells to nearby cells. LayerNorm acts on each edge, so
padding and batch size do not affect normalization. True marks a real cell.
"""

import math

import torch
from torch import nn


def _neighbors(points, valid, k, *, angular):
    """Find neighbors and use canonical cell order to break equal distances."""
    batch, count, _ = points.shape
    neighbors = min(k, max(count - 1, 0))
    if neighbors == 0:
        shape = (batch, count, 0)
        return (torch.empty(shape, dtype=torch.long, device=points.device),
                torch.empty(shape, dtype=torch.bool, device=points.device))
    with torch.no_grad():
        if points.dtype in (torch.float16, torch.bfloat16):
            points = points.float()
        if angular:
            eta, phi = points.unbind(dim=-1)
            deta = eta.unsqueeze(2) - eta.unsqueeze(1)
            dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
            dphi.add_(math.pi).remainder_(2 * math.pi).sub_(math.pi)
            distance = deta.square_().add_(dphi.square_())
        else:
            # Use the same distance calculation for short and padded batches.
            distance = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
        distance.masked_fill_(~valid.unsqueeze(1), torch.inf)
        distance.diagonal(dim1=1, dim2=2).fill_(torch.inf)
        indices = distance.argsort(dim=-1, stable=True)[..., :neighbors].contiguous()
        nearest = distance.gather(-1, indices)
        return indices, valid.unsqueeze(-1) & torch.isfinite(nearest)


def _canonical_cell_order(cells, valid):
    """Give equal-distance neighbors a tie-break independent of storage order."""
    # Cells can share an angular position in different calorimeter layers.
    # Sort valid cells by geometry, then energy and time. If every feature
    # agrees, exchanging the cells cannot change the information in an edge.
    geometry_then_measurements = (4, 5, 2, 3, 1, 0)
    with torch.no_grad():
        order = torch.arange(cells.shape[1], device=cells.device).expand(cells.shape[0], -1)
        for feature in reversed(geometry_then_measurements):
            ranks = cells[..., feature].gather(1, order).argsort(dim=1, stable=True)
            order = order.gather(1, ranks)
        ranks = (~valid.gather(1, order)).to(torch.uint8).argsort(dim=1, stable=True)
        return order.gather(1, ranks)


class _EdgeConv(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        layers = []
        width = 2 * input_width
        for _ in range(3):
            layers += [nn.Linear(width, output_width), nn.LayerNorm(output_width), nn.ReLU()]
            width = output_width
        self.edge_mlp = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.Linear(input_width, output_width, bias=False), nn.LayerNorm(output_width),
        )
        self.activation = nn.ReLU()

    def forward(self, features, valid, indices, valid_edges):
        batch, count, _ = features.shape
        shortcut = self.shortcut(features)
        batch_idx, center_idx, slot_idx = valid_edges.nonzero(as_tuple=True)
        if len(batch_idx):
            neighbor_idx = indices[batch_idx, center_idx, slot_idx]
            centers = features[batch_idx, center_idx]
            neighbors = features[batch_idx, neighbor_idx]
            edges = self.edge_mlp(torch.cat((centers, neighbors - centers), dim=-1))
            pooled = edges.new_zeros((batch * count, edges.shape[-1]))
            pooled.index_add_(0, batch_idx * count + center_idx, edges)
            neighbor_count = valid_edges.sum(dim=-1, keepdim=True).clamp_min(1)
            shortcut = shortcut + pooled.reshape_as(shortcut) / neighbor_count
        return self.activation(shortcut).masked_fill(~valid.unsqueeze(-1), 0)


class ParticleNetLiteCells(nn.Module):
    """Two EdgeConv blocks with k=7 and a jet-level classifier."""

    k = 7

    def __init__(self, jet_dim, cell_dim, angular_scale=(1., 1.)):
        super().__init__()
        if cell_dim != 6:
            raise ValueError("ParticleNet-Lite expects the six standard cell features")
        scale = torch.as_tensor(angular_scale, dtype=torch.float32)
        if scale.shape != (2,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("angular_scale must contain positive finite eta and phi scales")
        self.register_buffer("angular_scale", scale.clone())
        self.edge_conv_1 = _EdgeConv(cell_dim, 32)
        self.edge_conv_2 = _EdgeConv(32, 64)
        self.classifier = nn.Sequential(
            nn.Linear(jet_dim + 64, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1),
        )

    def forward(self, jets, cells, mask):
        batch, count, _ = cells.shape
        if count == 0:
            pooled = jets.new_zeros((batch, 64))
        else:
            valid = mask.bool()
            features = cells.masked_fill(~valid.unsqueeze(-1), 0)
            order = _canonical_cell_order(features, valid)
            valid = valid.gather(1, order)
            features = features.gather(1, order.unsqueeze(-1).expand_as(features))
            # Undo angle scaling. A common offset cancels in angle differences.
            angles = features[..., 4:6] * self.angular_scale
            graph = _neighbors(angles, valid, self.k, angular=True)
            features = self.edge_conv_1(features, valid, *graph)
            graph = _neighbors(features, valid, self.k, angular=False)
            features = self.edge_conv_2(features, valid, *graph)
            pooled = features.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        return self.classifier(torch.cat((jets, pooled), dim=-1)).squeeze(-1)
