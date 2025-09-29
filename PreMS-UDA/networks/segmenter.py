from Ours_v1_new.networks.common_modules import *


class SegmenterEncoder(nn.Module):
    def __init__(self, in_channels, n_channels, sample_nums, norm_type="gn", act_type="relu"):
        super(SegmenterEncoder, self).__init__()

        self.sample_nums = sample_nums
        channel_list = [n_channels * (2 ** i) for i in range(sample_nums + 1)]

        self.encode_module = nn.ModuleList()
        encode_block1 = nn.Sequential(
            nn.Conv2d(in_channels, channel_list[0], kernel_size=3, stride=1, padding=1),
            norm_layer(channel_list[0], norm_type),
            act_layer(act_type),

            nn.Conv2d(channel_list[0], channel_list[0], kernel_size=3, stride=1, padding=1),
            norm_layer(channel_list[0], norm_type),
            act_layer(act_type),

            nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, stride=2, padding=1),
            norm_layer(channel_list[1], norm_type),
            act_layer(act_type)
        )
        self.encode_module.append(encode_block1)

        encode_block2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[1], kernel_size=3, stride=1, padding=1),
            norm_layer(channel_list[1], norm_type),
            act_layer(act_type),

            nn.Conv2d(channel_list[1], channel_list[1], kernel_size=3, stride=1, padding=1),
            norm_layer(channel_list[1], norm_type),
            act_layer(act_type),

            nn.Conv2d(channel_list[1], channel_list[2], kernel_size=3, stride=2, padding=1),
            norm_layer(channel_list[2], norm_type),
            act_layer(act_type)
        )
        self.encode_module.append(encode_block2)

        encode_block3 = nn.Sequential(
            nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, stride=1, padding=1),
            norm_layer(channel_list[2], norm_type),
            act_layer(act_type),

            nn.Conv2d(channel_list[2], channel_list[2], kernel_size=3, stride=1, padding=1),
            norm_layer(channel_list[2], norm_type),
            act_layer(act_type),

            nn.Conv2d(channel_list[2], channel_list[3], kernel_size=3, stride=2, padding=1),
            norm_layer(channel_list[3], norm_type),
            act_layer(act_type)
        )
        self.encode_module.append(encode_block3)
        assert len(self.encode_module) == sample_nums

        self.residual_blocks = nn.Sequential(
            ResidualBlock(channel_list[-1], channel_list[-1], norm_type, act_type),
            ResidualBlock(channel_list[-1], channel_list[-1], norm_type, act_type),
            ResidualBlock(channel_list[-1], channel_list[-1], norm_type, act_type)
        )

        # !!!!!!!!!!
        self.normalize = nn.Sequential(
            norm_layer(channel_list[-1], "in_2d"),
            act_layer(act_type)
        )

    def forward(self, x):
        for i in range(self.sample_nums):
            x = self.encode_module[i](x)
        x = self.residual_blocks(x)
        x = self.normalize(x)

        return x


class SegmenterDecoder(nn.Module):
    def __init__(self, out_channels, n_channels, class_nums, sample_nums, norm_type="gn", act_type='relu'):
        super(SegmenterDecoder, self).__init__()

        self.out_channels = out_channels
        self.class_nums = class_nums
        self.sample_nums = sample_nums
        channel_list = [n_channels // (2 ** i) for i in range(sample_nums + 1)]

        self.residual_blocks = nn.Sequential(
            ResidualBlock(n_channels, n_channels, norm_type, act_type),
            ResidualBlock(n_channels, n_channels, norm_type, act_type),
            ResidualBlock(n_channels, n_channels, norm_type, act_type)
        )

        self.decode_module = nn.ModuleList()
        for i in range(sample_nums):
            decode_block = nn.Sequential(
                nn.ConvTranspose2d(
                    channel_list[i], channel_list[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(channel_list[i + 1], norm_type),
                act_layer(act_type),

                nn.Conv2d(channel_list[i + 1], channel_list[i + 1], kernel_size=3, stride=1, padding=1),
                norm_layer(channel_list[i + 1], norm_type),
                act_layer(act_type),

                nn.Conv2d(channel_list[i + 1], channel_list[i + 1], kernel_size=3, stride=1, padding=1),
                norm_layer(channel_list[i + 1], norm_type),
                act_layer(act_type)
            )
            self.decode_module.append(decode_block)

        self.out_conv = nn.Conv2d(
            channel_list[-1], class_nums * out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.residual_blocks(x)
        for i in range(self.sample_nums):
            x = self.decode_module[i](x)
        x = self.out_conv(x)

        x = torch.reshape(x, (x.shape[0], self.class_nums, self.out_channels, x.shape[2], x.shape[3]))
        if not self.training:
            x = torch.softmax(x, dim=1)

        return x
