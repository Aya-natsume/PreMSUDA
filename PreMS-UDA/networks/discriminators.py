from Ours_v1_new.networks.common_modules import *


class ImageDiscriminator(nn.Module):
    def __init__(self, in_channels, n_channels, sample_nums, norm_type="gn", act_type="leaky"):
        super(ImageDiscriminator, self).__init__()

        self.sample_nums = sample_nums
        channel_list = [n_channels * (2 ** i) for i in range(sample_nums + 1)]

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(n_channels, norm_type),
            act_layer(act_type)
        )

        self.down_sample = nn.ModuleList()
        for i in range(sample_nums):
            down_block = nn.Sequential(
                nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size=3, stride=2, padding=1),
                norm_layer(channel_list[i + 1], norm_type),
                act_layer(act_type)
            )
            self.down_sample.append(down_block)

        self.out_conv = nn.Conv2d(channel_list[-1], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        for i in range(self.sample_nums):
            x = self.down_sample[i](x)
        x = self.out_conv(x)

        return x


class FeatureDiscriminator(nn.Module):
    def __init__(self, n_channels, norm_type="gn", act_type="leaky"):
        super(FeatureDiscriminator, self).__init__()

        self.residual_blocks = nn.Sequential(
            ResidualBlock(n_channels, n_channels, norm_type, act_type),
            ResidualBlock(n_channels, n_channels, norm_type, act_type),
            ResidualBlock(n_channels, n_channels, norm_type, act_type)
        )

        self.out_conv = nn.Conv2d(n_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.residual_blocks(x)
        x = self.out_conv(x)

        return x
