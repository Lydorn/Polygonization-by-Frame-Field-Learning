from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBackbone(nn.Module):
    def __init__(self, n_channels, n_hidden_base, no_padding=False):
        super(UNetBackbone, self).__init__()
        self.no_padding = no_padding
        self.inc = InConv(n_channels, n_hidden_base, no_padding)
        self.down1 = Down(n_hidden_base, n_hidden_base*2, no_padding)
        self.down2 = Down(n_hidden_base*2, n_hidden_base*4, no_padding)
        self.down3 = Down(n_hidden_base*4, n_hidden_base*8, no_padding)
        self.down4 = Down(n_hidden_base*8, n_hidden_base*16, no_padding)

        self.up1 = Up(n_hidden_base*16, n_hidden_base*8, n_hidden_base*8, no_padding)
        self.up2 = Up(n_hidden_base*8, n_hidden_base*4, n_hidden_base*4, no_padding)
        self.up3 = Up(n_hidden_base*4, n_hidden_base*2, n_hidden_base*2, no_padding)
        self.up4 = Up(n_hidden_base*2, n_hidden_base, n_hidden_base, no_padding)

    def forward(self, x):
        x0 = self.inc.forward(x)
        x1 = self.down1.forward(x0)
        x2 = self.down2.forward(x1)
        x3 = self.down3.forward(x2)
        y4 = self.down4.forward(x3)
        y3 = self.up1.forward(y4, x3)
        y2 = self.up2.forward(y3, x2)
        y1 = self.up3.forward(y2, x1)
        y0 = self.up4.forward(y1, x0)

        result = OrderedDict()
        result["out"] = y0

        return result


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, no_padding):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0 if no_padding else 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=0 if no_padding else 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, no_padding):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, no_padding)

    def forward(self, x):
        x = self.conv.forward(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, no_padding):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, no_padding)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, no_padding):
        super(Up, self).__init__()
        self.conv = DoubleConv(in_ch_1 + in_ch_2, out_ch, no_padding)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv.forward(x)
        return x
