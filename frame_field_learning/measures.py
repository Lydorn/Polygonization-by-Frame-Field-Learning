import torch


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
