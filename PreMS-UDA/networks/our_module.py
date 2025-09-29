import torch.nn.functional as F
from Ours_v1_new.networks.common_modules import *


class MultiScaleFeatureFusion(nn.Module):  # 3D
    def __init__(self, n_channels, n_slices, kernel_list: list, norm_type="gn", act_type="relu"):
        super(MultiScaleFeatureFusion, self).__init__()

        self.ms_time = len(kernel_list)
        self.n_slices = n_slices
        self.kernel_list = kernel_list
        self.padding_list = [(0, 0, 0, 0, (k - 1) // 2, k // 2) for k in kernel_list]

        self.fusion_block = nn.ModuleList()
        for i in range(self.ms_time):
            if i == self.ms_time - 1:
                out_channels = n_channels // 2 ** i
            else:
                out_channels = n_channels // 2 ** (i + 1)

            self.fusion_block.append(nn.Sequential(
                nn.Conv3d(n_channels, out_channels,
                          kernel_size=(kernel_list[i], 1, 1), stride=(1, 1, 1)),
                norm_layer(out_channels, norm_type),
                act_layer(act_type)
            ))

        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(n_channels * n_slices, n_channels * n_slices),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        b_n, c, h, w = x.shape
        b = b_n // self.n_slices
        x = torch.reshape(x, (b, self.n_slices, c, h, w))
        x = torch.permute(x, (0, 2, 1, 3, 4))

        fusion_feature_list = []
        for i in range(self.ms_time):
            x_ = F.pad(x, self.padding_list[i])
            fusion_feature = self.fusion_block[i](x_)
            fusion_feature_list.append(fusion_feature)

        ms_feature = torch.cat(fusion_feature_list, dim=1)
        ms_weight = self.pool(ms_feature)
        ms_weight = self.fc(ms_weight).unsqueeze(-1).unsqueeze(-1)
        ms_weight = torch.reshape(ms_weight, (b, c, self.n_slices, 1, 1))
        ms_weight = self.act(ms_weight)

        ms_feature = ms_feature * ms_weight + ms_feature

        ms_feature = torch.permute(ms_feature, (0, 2, 1, 3, 4))
        x = torch.reshape(ms_feature, (b_n, c, h, w))

        return x
