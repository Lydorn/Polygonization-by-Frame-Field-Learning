import torch
from torch.nn import functional as F

from torch_lydorn.torch.utils.complex import complex_mul, complex_abs_squared


def iou(y_pred, y_true, threshold):
    assert len(y_pred.shape) == len(y_true.shape) == 2, "Input tensor shapes should be (N, .)"
    mask_pred = threshold < y_pred
    mask_true = threshold < y_true
    intersection = torch.sum(mask_pred * mask_true, dim=-1)
    union = torch.sum(mask_pred + mask_true, dim=-1)
    r = intersection.float() / union.float()
    r[union == 0] = 1
    return r


def dice_loss(y_pred, y_true, smooth=1, eps=1e-7):
    """

    @param y_pred: (N, C, H, W)
    @param y_true: (N, C, H, W)
    @param smooth:
    @param eps:
    @return: (N, C)
    """
    numerator = 2 * torch.sum(y_true * y_pred, dim=(-1, -2))
    denominator = torch.sum(y_true, dim=(-1, -2)) + torch.sum(y_pred, dim=(-1, -2))
    return 1 - (numerator + smooth) / (denominator + smooth + eps)


def crossfield_align_error(c0, c2, z, complex_dim=-1):
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


def main():
    # import kornia
    # spatial_gradient_function = kornia.filters.SpatialGradient()
    #
    # image = torch.zeros((7, 7))
    # image[2:5, 2:5] = 1
    # print(image)
    #
    # grads = spatial_gradient_function(image[None, None, ...])[0, 0, ...] / 4
    # print(grads[0])
    # print(grads[1])

    y_true = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    y_pred = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    print(y_true.shape)
    print(y_pred.shape)
    r = iou(y_pred, y_true, threshold=0.5)
    print(r)


if __name__ == "__main__":
    main()
