import numpy as np

import torch
from torch.nn import functional as F

from torch_lydorn.torch.utils.complex import complex_mul, complex_sqrt, complex_abs_squared


def framefield_align_error(c0, c2, z, complex_dim=-1):
    assert c0.shape == c2.shape == z.shape, \
        "All inputs should have the same shape. Currently c0: {}, c2: {}, z: {}".format(c0.shape, c2.shape, z.shape)
    assert c0.shape[complex_dim] == c2.shape[complex_dim] == z.shape[complex_dim] == 2, \
        "All inputs should have their complex_dim size equal 2 (real and imag parts)"

    z_squared = complex_mul(z, z, complex_dim=complex_dim)
    z_pow_4 = complex_mul(z_squared, z_squared, complex_dim=complex_dim)
    # All tensors are assimilated as being complex so adding that way works (adding a scalar wouldn't work):
    f_z = z_pow_4 + complex_mul(c2, z_squared, complex_dim=complex_dim) + c0
    loss = complex_abs_squared(f_z, complex_dim)  # Square of the absolute value of f_z
    return loss


class LaplacianPenalty:
    def __init__(self, channels: int):
        self.channels = channels
        self.filter = torch.tensor([[0.5, 1.0, 0.5],
                                    [1.0, -6., 1.0],
                                    [0.5, 1.0, 0.5]]) / 12
        self.filter = self.filter[None, None, ...].expand(self.channels, -1, -1, -1)

    def laplacian_filter(self, tensor):
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        penalty_tensor = F.conv2d(tensor, self.filter.to(tensor.device), padding=1,
                                  groups=self.channels)
        # print("penalty_tensor min={}, max={}".format(penalty_tensor.min(), penalty_tensor.max()))
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        return torch.abs(penalty_tensor)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.laplacian_filter(tensor)


def c0c2_to_uv(c0c2: torch.Tensor) -> torch.Tensor:
    c0, c2 = torch.chunk(c0c2, 2, dim=1)
    c2_squared = complex_mul(c2, c2, complex_dim=1)
    c2_squared_minus_4c0 = c2_squared - 4 * c0
    sqrt_c2_squared_minus_4c0 = complex_sqrt(c2_squared_minus_4c0, complex_dim=1)
    u_squared = (c2 + sqrt_c2_squared_minus_4c0) / 2
    v_squared = (c2 - sqrt_c2_squared_minus_4c0) / 2
    uv_squared = torch.stack([u_squared, v_squared], dim=1)  # Shape (B, 'uv': 2, 'complex': 2, H, W)
    uv = complex_sqrt(uv_squared, complex_dim=2)
    return uv


def compute_closest_in_uv(directions: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """
    For each direction, compute if it is more aligned with {u, -u} (output 0) or {v, -v} (output 1).

    @param directions: Tensor of shape (N, 2)
    @param uv: Tensor of shape (N, 'uv': 2, 'complex': 2)
    @return: closest_in_uv of shape (N,) with the index in the 'uv' dimension of the closest vector in uv to direction
    """
    uv_dot_dir = torch.sum(uv * directions[:, None, :], dim=2)
    abs_uv_dot_dir = torch.abs(uv_dot_dir)

    closest_in_uv = torch.argmin(abs_uv_dot_dir, dim=1)

    return closest_in_uv


def detect_corners(polylines, u, v):
    def compute_direction_score(ij, edges, field_dir):
        values = field_dir[ij[:, 0], ij[:, 1]]
        edge_dot_dir = edges[:, 0] * values.real + edges[:, 1] * values.imag
        abs_edge_dot_dir = np.abs(edge_dot_dir)
        return abs_edge_dot_dir

    def compute_is_corner(points, left_edges, right_edges):
        if points.shape[0] == 0:
            return np.empty(0, dtype=np.bool)

        coords = np.round(points).astype(np.int)
        coords[:, 0] = np.clip(coords[:, 0], 0, u.shape[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, u.shape[1] - 1)
        left_u_score = compute_direction_score(coords, left_edges, u)
        left_v_score = compute_direction_score(coords, left_edges, v)
        right_u_score = compute_direction_score(coords, right_edges, u)
        right_v_score = compute_direction_score(coords, right_edges, v)

        left_is_u_aligned = left_v_score < left_u_score
        right_is_u_aligned = right_v_score < right_u_score

        return np.logical_xor(left_is_u_aligned, right_is_u_aligned)

    corner_masks = []
    for polyline in polylines:
        corner_mask = np.zeros(polyline.shape[0], dtype=np.bool)
        if np.max(np.abs(polyline[0] - polyline[-1])) < 1e-6:
            # Closed polyline
            left_edges = np.concatenate([polyline[-2:-1] - polyline[-1:], polyline[:-2] - polyline[1:-1]], axis=0)
            right_edges = polyline[1:] - polyline[:-1]
            corner_mask[:-1] = compute_is_corner(polyline[:-1, :], left_edges, right_edges)
            # left_edges and right_edges do not include the redundant last vertex, thus we have to do this assignment:
            corner_mask[-1] = corner_mask[0]
        else:
            # Open polyline
            corner_mask[0] = True
            corner_mask[-1] = True
            left_edges = polyline[:-2] - polyline[1:-1]
            right_edges = polyline[2:] - polyline[1:-1]
            corner_mask[1:-1] = compute_is_corner(polyline[1:-1, :], left_edges, right_edges)
        corner_masks.append(corner_mask)

    return corner_masks

