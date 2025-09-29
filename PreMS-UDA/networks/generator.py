from Ours_v1_new.taming.modules.diffusionmodules.model import *
from Ours_v1_new.taming.modules.vqvae.quantizer import *


class GeneratorEncoder(nn.Module):
    def __init__(self, in_channels, n_channels, sample_nums, norm_type="gn", act_type="relu"):
        super(GeneratorEncoder, self).__init__()

        self.encoder = Encoder(
            in_channels, n_channels, sample_nums, norm_type, act_type
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)

        return x


class GroupQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dims, beta=0.25, legacy=False):
        super(GroupQuantizer, self).__init__()

        self.embed_dims = embed_dims
        self.vector_quantizer = VectorQuantizerV2(n_e=n_embed, e_dim=embed_dims, beta=beta, legacy=legacy)

    def forward(self, x):
        with torch.no_grad():
            x_list = torch.split(x, self.embed_dims, dim=1)
            quant_list = []
            for x_ in x_list:
                quant, _, _ = self.vector_quantizer(x_)
                quant_list.append(quant)
            x = torch.cat(quant_list, dim=1)

        return x


class GeneratorDecoder(nn.Module):
    def __init__(self, out_channels, n_channels, sample_nums, norm_type="gn", act_type="relu"):
        super(GeneratorDecoder, self).__init__()

        self.decoder = Decoder(
            out_channels, n_channels, sample_nums, norm_type, act_type
        )

    def forward(self, x):
        x = self.decoder(x)
        x = torch.clamp(x, -1., 1.)

        return x
