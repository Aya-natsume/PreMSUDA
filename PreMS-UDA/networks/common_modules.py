import torch
import torch.nn as nn
import timm


class EncodeConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="gn", act_type="relu"):
        super(EncodeConv, self).__init__()

        self.encode_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels, norm_type),
            act_layer(act_type),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels, norm_type),
            act_layer(act_type)
        )

    def forward(self, x):
        x = self.encode_conv(x)

        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, sample_type="conv", norm_type="gn", act_type="relu"):
        super(DownSample, self).__init__()

        assert sample_type.lower() in ["pool", "conv"]
        if sample_type == "pool":
            self.down_sample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),

                EncodeConv(in_channels, out_channels, norm_type, act_type)
            )
        else:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels, norm_type),
                act_layer(act_type),

                EncodeConv(out_channels, out_channels, norm_type, act_type)
            )

    def forward(self, x):
        x = self.down_sample(x)

        return x


class UpSampleV1(nn.Module):
    def __init__(self, in_channels, out_channels, sample_type="conv", norm_type="gn", act_type="relu"):
        super(UpSampleV1, self).__init__()

        assert sample_type.lower() in ["up", "conv"]
        if sample_type.lower() == "up":
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.encode_conv = EncodeConv(
                in_channels + out_channels, out_channels, norm_type, act_type
            )
        else:
            self.up_sample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(out_channels, norm_type),
                act_layer(act_type)
            )
            self.encode_conv = EncodeConv(in_channels, out_channels, norm_type, act_type)

    def forward(self, x, x_):
        x = self.up_sample(x)
        x = torch.cat([x, x_], dim=1)
        x = self.encode_conv(x)

        return x


class UpSampleV2(nn.Module):
    def __init__(self, in_channels, out_channels, sample_type="conv", norm_type="gn", act_type="relu"):
        super(UpSampleV2, self).__init__()

        assert sample_type.lower() in ["up", "conv"]
        if sample_type.lower() == "up":
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.encode_conv = EncodeConv(in_channels, out_channels, norm_type, act_type)
        else:
            self.up_sample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(out_channels, norm_type),
                act_layer(act_type)
            )
            self.encode_conv = EncodeConv(out_channels, out_channels, norm_type, act_type)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.encode_conv(x)

        return x


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
