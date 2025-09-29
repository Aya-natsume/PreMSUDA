import torch.nn as nn
import timm


def norm_layer(n_channels, norm_type="gn"):
    if norm_type.lower() == "gn":
        return nn.GroupNorm(num_groups=32 if n_channels % 32 == 0 else n_channels, num_channels=n_channels)
    elif norm_type.lower() == "bn_2d":
        return nn.BatchNorm2d(n_channels)
    elif norm_type.lower() == "bn_3d":
        return nn.BatchNorm3d(n_channels)
    elif norm_type.lower() == "in_2d":
        return nn.InstanceNorm2d(n_channels)
    elif norm_type.lower() == "in_3d":
        return nn.InstanceNorm3d(n_channels)
    elif norm_type.lower() == "ln":
        return nn.LayerNorm(n_channels)
    else:
        print("Unsupported normalization")
        return 0


def act_layer(act_type="relu"):
    if act_type.lower() == "relu":
        return nn.ReLU()
    elif act_type.lower() == "leaky":
        return nn.LeakyReLU(0.2)
    elif act_type.lower() == "swish":
        return timm.layers.activations.Swish()
    else:
        print("Unsupported activation")
        return 0


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="gn", act_type="relu"):
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels, norm_type),
            act_layer(act_type),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels, norm_type),
        )

        self.act = act_layer(act_type)

    def forward(self, x):
        x = x + self.residual_block(x)
        x = self.act(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, n_channels, sample_nums, norm_type="gn", act_type="relu"):
        super(Encoder, self).__init__()

        self.sample_nums = sample_nums
        channel_list = [n_channels * (2 ** i) for i in range(sample_nums + 1)]

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(n_channels, norm_type),
            act_layer(act_type),
            ResidualBlock(n_channels, n_channels, norm_type, act_type),
            ResidualBlock(n_channels, n_channels, norm_type, act_type)
        )

        self.encode_module = nn.ModuleList()
        for i in range(sample_nums):
            encode_block = nn.Sequential(
                nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size=3, stride=2, padding=1),
                norm_layer(channel_list[i + 1], norm_type),
                act_layer(act_type),
                ResidualBlock(
                    channel_list[i + 1], channel_list[i + 1], norm_type, act_type
                ),
                ResidualBlock(
                    channel_list[i + 1], channel_list[i + 1], norm_type, act_type
                )
            )
            self.encode_module.append(encode_block)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(channel_list[-1], channel_list[-1], norm_type, act_type),
            ResidualBlock(channel_list[-1], channel_list[-1], norm_type, act_type)
        )

    def forward(self, x):
        x = self.in_conv(x)
        for i in range(self.sample_nums):
            x = self.encode_module[i](x)
        x = self.residual_blocks(x)

        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, n_channels, sample_nums, norm_type="gn", act_type="relu"):
        super(Decoder, self).__init__()

        self.sample_nums = sample_nums
        channel_list = [n_channels // (2 ** i) for i in range(sample_nums + 1)]

        self.residual_blocks = nn.Sequential(
            ResidualBlock(n_channels, n_channels, norm_type, act_type),
            ResidualBlock(n_channels, n_channels, norm_type, act_type)
        )

        self.decode_module = nn.ModuleList()
        for i in range(sample_nums):
            decode_block = nn.Sequential(
                ResidualBlock(channel_list[i], channel_list[i], norm_type, act_type),
                ResidualBlock(channel_list[i], channel_list[i], norm_type, act_type),
                nn.ConvTranspose2d(channel_list[i], channel_list[i + 1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(channel_list[i + 1], norm_type),
                act_layer(act_type)
            )
            self.decode_module.append(decode_block)

        self.out_conv = nn.Sequential(
            ResidualBlock(channel_list[-1], channel_list[-1], norm_type, act_type),
            ResidualBlock(channel_list[-1], channel_list[-1], norm_type, act_type),
            nn.Conv2d(channel_list[-1], out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.residual_blocks(x)
        for i in range(self.sample_nums):
            x = self.decode_module[i](x)
        x = self.out_conv(x)

        return x
