import torch

from torch_lydorn.torch.utils.complex import complex_mul, complex_sqrt


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

