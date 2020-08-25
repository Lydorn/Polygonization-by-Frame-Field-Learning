from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
import torch
import torch.utils.checkpoint

# from pytorch_memlab import profile, profile_every


def humanbytes(B):
    'Return the given bytes as a human friendly KB, MB, GB, or TB string'
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


def get_preact_conv(in_channels, out_channels, kernel_size=3, padding=1, dropout_2d=0.2):
    block = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.Dropout2d(dropout_2d)
    )
    return block


def _dense_layer_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_2d=0.2, efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
        self.dropout_2d = dropout_2d
        self.efficient = efficient

    def forward(self, *prev_features):
        dense_layer_function = _dense_layer_function_factory(self.norm, self.relu, self.conv)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            new_features = torch.utils.checkpoint.checkpoint(dense_layer_function, *prev_features)
        else:
            new_features = dense_layer_function(*prev_features)
        if 0 < self.dropout_2d:
            new_features = F.dropout2d(new_features, p=self.dropout_2d, training=self.training)
        return new_features


class SELayer(nn.Module):
    def __init__(self, in_channels, ratio):
        super(SELayer, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),  # Prepare for fully-connected layers
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        excitation = self.block(x)
        x *= excitation[:, :, None, None]
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, n_layers, growth_rate, dropout_2d, return_only_new=False, efficient=False):
        super(DenseBlock, self).__init__()
        assert 0 < n_layers, "n_layers should be at least 1"
        self.in_channels = in_channels
        self.return_only_new = return_only_new

        channels = in_channels
        self.layers = torch.nn.ModuleList()
        for j in range(n_layers):
            # Compute new feature maps
            layer = DenseLayer(channels, growth_rate, dropout_2d=dropout_2d, efficient=efficient)
            self.layers.append(layer)
            channels += growth_rate

        if return_only_new:
            se_layer_in_channel = channels - in_channels  # Remove input, only keep new features
        else:
            se_layer_in_channel = channels
        self.se_layer = SELayer(se_layer_in_channel, ratio=1)

    # @profile_every(1)
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(*features)
            features.append(new_features)

        if self.return_only_new:
            features = features[1:]

        features = torch.cat(features, 1)
        features = self.se_layer(features)

        return features


def get_transition_down(in_channels, out_channels, dropout_2d=0.2):
    block = nn.Sequential(
        get_preact_conv(in_channels, out_channels, kernel_size=1, padding=0, dropout_2d=dropout_2d),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return block


def cat_non_matching(x1, x2):
    diffY = x1.size()[2] - x2.size()[2]
    diffX = x1.size()[3] - x2.size()[3]

    x2 = F.pad(x2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

    # for padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

    x = torch.cat([x1, x2], dim=1)
    return x


class TransitionUp(nn.Module):
    def __init__(self, in_channels, n_filters_keep):
        super(TransitionUp, self).__init__()
        self.conv_transpose_2d = nn.ConvTranspose2d(in_channels, n_filters_keep, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip_connection):
        x = self.conv_transpose_2d(x)
        x = cat_non_matching(x, skip_connection)
        return x


class ICTNetBackbone(nn.Module):
    """
    ICTNet model: https://theictlab.org/lp/2019ICTNet.
    """
    def __init__(self, preset_model='FC-DenseNet56', in_channels=3, out_channels=2, n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_2d=0.2, efficient=False):
        super().__init__()

        # --- Handle args
        if preset_model == 'FC-DenseNet56':
            n_pool = 5
            growth_rate = 12
            n_layers_per_block = 4
        elif preset_model == 'FC-DenseNet67':
            n_pool = 5
            growth_rate = 16
            n_layers_per_block = 5
        elif preset_model == 'FC-DenseNet103':
            n_pool = 5
            growth_rate = 16
            n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
        else:
            n_pool = n_pool
            growth_rate = growth_rate
            n_layers_per_block = n_layers_per_block

        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * n_pool + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
        else:
            raise ValueError

        # --- Instantiate layers
        self.first_conv = nn.Conv2d(in_channels, n_filters_first_conv, 3, padding=1)

        # Downsampling path
        channels = n_filters_first_conv
        self.down_dense_blocks = torch.nn.ModuleList()
        self.transition_downs = torch.nn.ModuleList()
        skip_connection_channels = []
        for i in range(n_pool):
            # Dense Block
            self.down_dense_blocks.append(DenseBlock(in_channels=channels, n_layers=n_layers_per_block[i], growth_rate=growth_rate, dropout_2d=dropout_2d, return_only_new=False, efficient=efficient))
            channels += growth_rate * n_layers_per_block[i]
            skip_connection_channels.append(channels)
            # Transition Down
            self.transition_downs.append(get_transition_down(in_channels=channels, out_channels=channels, dropout_2d=dropout_2d))

        # Bottleneck Dense Block
        self.bottleneck_dense_block = DenseBlock(in_channels=channels, n_layers=n_layers_per_block[n_pool], growth_rate=growth_rate, dropout_2d=dropout_2d, return_only_new=True, efficient=efficient)
        up_in_channels = n_layers_per_block[n_pool] * growth_rate  # We will only upsample the new feature maps

        # Upsampling path
        self.transition_ups = torch.nn.ModuleList()
        self.up_dense_blocks = torch.nn.ModuleList()
        for i in range(n_pool):
            # Transition Up (Upsampling + concatenation with the skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            self.transition_ups.append(TransitionUp(in_channels=up_in_channels, n_filters_keep=n_filters_keep))
            up_out_channels = skip_connection_channels[n_pool - i - 1] + n_filters_keep  # After concatenation

            # Dense Block
            # We will only upsample the new feature maps
            self.up_dense_blocks.append(
                DenseBlock(in_channels=up_out_channels, n_layers=n_layers_per_block[n_pool + i + 1], growth_rate=growth_rate,
                           dropout_2d=dropout_2d, return_only_new=True, efficient=efficient))
            up_in_channels = growth_rate * n_layers_per_block[n_pool + i + 1]  # We will only upsample the new feature maps

        # Last layer
        self.final_conv = nn.Conv2d(up_in_channels, out_channels, 1, padding=0)

    # @profile
    def forward(self, x):
        stack = self.first_conv(x)

        skip_connection_list = []
        # print(humanbytes(torch.cuda.memory_allocated()))
        for down_dense_block, transition_down in zip(self.down_dense_blocks, self.transition_downs):
            # Dense Block
            stack = down_dense_block(stack)

            # At the end of the dense block, the current stack is stored in the skip_connections list
            skip_connection_list.append(stack)

            # Transition Down
            stack = transition_down(stack)
        # print(humanbytes(torch.cuda.memory_allocated()))

        skip_connection_list = skip_connection_list[::-1]

        # Bottleneck Dense Block
        # We will only upsample the new feature maps
        stack = self.bottleneck_dense_block(stack)

        # Upsampling path
        # print(humanbytes(torch.cuda.memory_allocated()))
        for transition_up, up_dense_block, skip_connection in zip(self.transition_ups, self.up_dense_blocks, skip_connection_list):
            # Transition Up ( Upsampling + concatenation with the skip connection)
            stack = transition_up(stack, skip_connection)

            # Dense Block
            # We will only upsample the new feature maps
            stack = up_dense_block(stack)
        # print(humanbytes(torch.cuda.memory_allocated()))

        # Final conv
        stack = self.final_conv(stack)

        result = OrderedDict()
        result["out"] = stack

        # print(humanbytes(torch.cuda.memory_allocated()))

        return result


def count_trainable_params(model):
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            count += param.numel()
    return count


def main():
    device = "cuda"
    b = 2
    c = 3
    h = 512
    w = 512
    features = 32

    # Init input
    x = torch.rand((b, c, h, w), device=device)
    print("x: ", x.shape, x.min().item(), x.max().item())

    # # Test SELayer
    # print("--- Test SELayer:")
    # se_layer = SELayer(in_channels=c, ratio=1)
    # y = se_layer(x)
    # print("y: ", y.shape)
    # print("------")

    # # Test DenseBlock
    # print("--- Test DenseBlock:")
    # dense_block = DenseBlock(in_channels=c, n_layers=5, growth_rate=16, dropout_2d=0.2, path_type="down")
    # y, new_y = dense_block(x)
    # print("y: ", y.shape)
    # print("new_y: ", new_y.shape)
    # print("------")

    # # Test transition_down
    # print("--- Test transition_down:")
    # transition_down = get_transition_down(in_channels=c, out_channels=features, dropout_2d=0.2)
    # x_down = transition_down(x)
    # print("x_down: ", x_down.shape)
    # print("------")
    #
    # # Test TransitionUp
    # print("--- Test TransitionUp:")
    # transition_up = TransitionUp(in_channels=features, n_filters_keep=features//2)
    # y = transition_up(x_down, x)
    # print("y: ", y.shape)
    # print("------")

    # Test ICTNetBackboneICTNetBackboneTest SELayer:")
    backbone = ICTNetBackbone(out_channels=features, preset_model="FC-DenseNet103", dropout_2d=0.0, efficient=True)
    print("ICTNetBackbone has {} trainable params".format(count_trainable_params(backbone)))
    # print(backbone)
    backbone.to(device)
    result = backbone(x)
    y = result["out"]
    print("y: ", y.shape)
    print("------")

    print("Back-prop:")
    loss = torch.sum(y)
    loss.backward()


if __name__ == "__main__":
    main()



















