# ---#
#
# Initial code from https://github.com/neptune-ai/open-solution-mapping-challenge/blob/master/src/unet_models.py
#
# ---
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
import torch
import torchvision


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        print("ConvRelu:", x.shape, x.min().item(), x.max().item())
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=True),
                nn.BatchNorm2d(middle_channels),
                nn.ELU(),
                nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU()
            )

    def forward(self, x):
        return self.block(x)


def cat_non_matching(x1, x2):
    diffY = x1.size()[2] - x2.size()[2]
    diffX = x1.size()[3] - x2.size()[3]

    x2 = F.pad(x2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

    # for padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

    x = torch.cat([x1, x2], dim=1)
    return x


class UNetResNetBackbone(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.
    """

    def __init__(self, encoder_depth, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)

    def forward(self, x):
        # print("x:", x.shape, x.min().item(), x.max().item())
        conv1 = self.conv1(x)
        # print("conv1:", conv1.shape, conv1.min().item(), conv1.max().item())
        conv2 = self.conv2(conv1)
        # print("conv2:", conv2.shape, conv2.min().item(), conv2.max().item())
        conv3 = self.conv3(conv2)
        # print("conv3:", conv3.shape, conv3.min().item(), conv3.max().item())
        conv4 = self.conv4(conv3)
        # print("conv4:", conv4.shape, conv4.min().item(), conv4.max().item())
        conv5 = self.conv5(conv4)
        # print("conv5:", conv5.shape, conv5.min().item(), conv5.max().item())

        pool = self.pool(conv5)
        # print("pool:", pool.shape, pool.min().item(), pool.max().item())
        center = self.center(pool)
        # print("center:", center.shape, center.min().item(), center.max().item())

        dec5 = self.dec5(cat_non_matching(conv5, center))
        # print("center:", center.shape, center.min().item(), center.max().item())

        dec4 = self.dec4(cat_non_matching(conv4, dec5))
        # print("dec4:", dec4.shape, dec4.min().item(), dec4.max().item())
        dec3 = self.dec3(cat_non_matching(conv3, dec4))
        # print("dec3:", dec3.shape, dec3.min().item(), dec3.max().item())
        dec2 = self.dec2(cat_non_matching(conv2, dec3))
        # print("dec2:", dec2.shape, dec2.min().item(), dec2.max().item())
        dec1 = self.dec1(dec2)
        # print("dec1:", dec1.shape, dec1.min().item(), dec1.max().item())

        y = F.dropout2d(dec1, p=self.dropout_2d)
        # print("y:", y.shape, y.min().item(), y.max().item())

        result = OrderedDict()
        result["out"] = y

        return result
