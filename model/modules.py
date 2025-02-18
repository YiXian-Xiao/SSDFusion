from collections import namedtuple

import torch
from torchvision.transforms import functional as TVF
from mmseg.models import BaseSegmentor
from torch import nn
from torch import fft
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from torch.nn import LayerNorm, BatchNorm2d, UpsamplingBilinear2d

from utils.constants import eps
from .layers import *
from mmseg.models.backbones.mscan import MSCABlock
from mmseg.models.backbones.mscan import OverlapPatchEmbed as PatchEmbedMSCAN


class DFF(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.channels = input_channels
        self.conv_1 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=1)
        self.norm = nn.BatchNorm2d(input_channels * 2)
        self.conv_2 = nn.Conv2d(2, 1, kernel_size=7, padding='same')

    def forward(self, x):
        x = fft.rfft2(x)

        fx = torch.view_as_real(x)

        fx = torch.cat([fx[:, :, :, :, 0], fx[:, :, :, :, 1]], dim=1)
        fx = self.conv_1(fx)
        fx = self.norm(fx)
        fx = F.relu(fx, True)

        max = torch.amax(fx, dim=1, keepdim=True)
        avg = torch.mean(fx, dim=1, keepdim=True)
        filter = F.sigmoid(self.conv_2(torch.cat([max, avg], dim=1)))

        fx = filter * fx  # Inplace mul not allowed here

        fx = fx.unsqueeze(-1)
        fx = torch.cat([fx[:, :self.channels, :, :, :], fx[:, self.channels:, :, :, :]], dim=4)

        return fft.irfft2(torch.view_as_complex(fx))


class FeatureRefiner(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 drop_path_rate=(0., 0.), downsample=True, num_blocks=1,
                 mlp_ratio=2):
        super().__init__()

        in_channels *= 2
        out_channels *= 2

        if hidden_channels is None:
            hidden_channels = out_channels

        self.patch_embed = PatchEmbedMSCAN(in_channels=in_channels, embed_dim=hidden_channels, patch_size=5,
                                           stride=2) if downsample \
            else (PatchEmbedMSCAN(in_channels=in_channels, embed_dim=hidden_channels, patch_size=3, stride=1))

        if not isinstance(drop_path_rate, tuple) or not isinstance(drop_path_rate, list):
            drop_path_rate = (0., drop_path_rate)

        dpr = [
            x.item() for x in torch.linspace(drop_path_rate[0], drop_path_rate[1], num_blocks)
        ] if num_blocks > 1 else [drop_path_rate[1]]

        self.extractor = nn.ModuleList(
            [MSCABlock(hidden_channels, drop_path=dpr[k], mlp_ratio=mlp_ratio) for k in range(num_blocks)]
        )
        self.norm = BatchNorm2d(hidden_channels)

        self.conv_2 = ConvLayer(hidden_channels, out_channels, activation=nn.Tanh(), kernel_size=1, padding='same')
        #  self.dff = DFF(out_channels)

        # TODO: Examine the feature refiner design

    def forward(self, feat_ir, feat_vi):
        B = feat_ir.shape[0]

        feat = torch.cat([feat_ir, feat_vi], dim=1)

        feat, H, W = self.patch_embed(feat)

        for extractor in self.extractor:
            feat = extractor(feat, H, W)

        feat = self.norm(feat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        feat = self.conv_2(feat)  # self.dff(self.conv_2(feat))

        feat_ir, feat_vi = torch.chunk(feat, 2, dim=1)

        return feat_ir, feat_vi


class FeatureFusor(nn.Module):
    def __init__(self, base_channels, detail_channels):
        super().__init__()
        fused_channels = (base_channels + detail_channels) // 2
        self.base_fusor = BaseFeatureExtraction(base_channels, 8)
        self.detail_fusor = DetailFeatureExtraction(detail_channels, 1)
        self.conv_fusor = nn.Conv2d(fused_channels, fused_channels, 1, padding='same')
        self.fusor = DRDB(3, fused_channels, 32)
        self.spade = SPADE(base_channels, detail_channels, 64)

    def forward(self, base_feat, detail_feat):
        base_feat = self.base_fusor(base_feat)
        detail_feat = self.detail_fusor(detail_feat)

        fused_feat = self.spade(base_feat, detail_feat)
        fused_feat = self.fusor(self.conv_fusor(fused_feat))

        return base_feat, detail_feat, fused_feat


class FeatureDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.eca = ECA_NS(in_channels)
        self.reduce_channel = nn.Conv2d(in_channels, out_channels, 1)
        self.fusor = DRDB(3, out_channels, 32)

    def forward(self, feat):
        feat = self.eca(feat)
        feat = self.reduce_channel(feat)
        return self.fusor(feat)


class FeatureExtractor(nn.Module):

    def __init__(self, in_channels, base_channels, detail_channels):
        super().__init__()
        # TODO: Examine the effect of these conv projections
        self.conv_base = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        self.conv_detail = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        self.base_extractor = BaseFeatureExtraction(base_channels, 8)
        self.detail_extractor = DetailFeatureExtraction(detail_channels, 3)

    def forward(self, feat):
        base_feat = self.base_extractor(self.conv_base(feat))
        detail_feat = self.detail_extractor(self.conv_detail(feat))

        return base_feat, detail_feat, feat


class FusionModule(nn.Module):
    Outputs = namedtuple('Outputs', ['base_feat', 'detail_feat', 'fused_feat',
                                     'base_feat_ir', 'base_feat_vi',
                                     'detail_feat_ir', 'detail_feat_vi',
                                     'fused_feat_ir', 'fused_feat_vi'])  # for stage 1

    def __init__(self, input_channels, base_channels, detail_channels):
        super().__init__()
        self.extractor_ir = FeatureExtractor(input_channels, base_channels, detail_channels)
        self.extractor_vi = FeatureExtractor(input_channels, base_channels, detail_channels)
        self.eca_base = ECA_NS(base_channels)
        self.eca_detail = ECA_NS(detail_channels)
        self.fusor = FeatureFusor(base_channels, detail_channels)
        self.stage = 2

    def forward(self, feat_ir, feat_vi) -> Outputs:
        base_feat_ir, detail_feat_ir, feat_ir = self.extractor_ir(feat_ir)
        base_feat_vi, detail_feat_vi, feat_vi = self.extractor_vi(feat_vi)

        if self.stage == 1:
            base_feat, detail_feat, fused_feat_vi = self.fusor(self.eca_base(base_feat_vi),
                                                               self.eca_detail(detail_feat_vi))
            base_feat, detail_feat, fused_feat_ir = self.fusor(self.eca_base(base_feat_ir),
                                                               self.eca_detail(detail_feat_ir))

            output = FusionModule.Outputs(None, None, None,
                                          base_feat_ir, base_feat_vi,
                                          detail_feat_ir, detail_feat_vi,
                                          fused_feat_ir, fused_feat_vi)

        elif self.stage == 2:
            base_feat, detail_feat, fused_feat = self.fusor(
                self.eca_base(base_feat_vi) + self.eca_base(base_feat_ir),
                self.eca_detail(detail_feat_vi) + self.eca_detail(detail_feat_ir))

            output = FusionModule.Outputs(base_feat, detail_feat, fused_feat,
                                          base_feat_ir, base_feat_vi,
                                          detail_feat_ir, detail_feat_vi,
                                          None, None)

        else:
            output = FusionModule.Outputs(None, None, None,
                                          base_feat_ir, base_feat_vi,
                                          detail_feat_ir, detail_feat_vi,
                                          None, None)

        return output


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = ChannelWiseLayerNorm(dim)
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = ChannelWiseLayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, in_channels):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        detail_channels = in_channels // 2
        self.theta_phi = InvertedResidualBlock(inp=detail_channels, oup=detail_channels, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=detail_channels, oup=detail_channels, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=detail_channels, oup=detail_channels, expand_ratio=2)
        self.shffleconv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, in_channels, num_layers=1):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(in_channels) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = torch.chunk(x, 2, dim=1)
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class MetaFeatureExtractor(nn.Module):
    def __init__(self, in_channels: tuple[int, int], out_channels, hidden_channels=None, drop_path_rate=0.):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.patch_embed = PatchEmbedMSCAN(in_channels=in_channels[0], embed_dim=hidden_channels, patch_size=5, stride=2)
        self.conv_1 = MSCABlock(hidden_channels, drop_path=drop_path_rate, mlp_ratio=2)
        self.norm = BatchNorm2d(hidden_channels)
        self.conv_2 = ConvLayer(hidden_channels, hidden_channels, kernel_size=1, padding='same')
        self.up = UpsamplingBilinear2d(scale_factor=2)
        #  self.dff = DFF(out_channels)

        self.extractor_seg = ConvNeXtBlock(in_channels[1], mlp_ratio=2)

        self.trans_1 = TransBlock((in_channels[1], hidden_channels), kernel_size=3, padding=1)
        self.fusor = DRDB(3, hidden_channels * 2, 32)

        self.trans_2 = TransBlock((hidden_channels * 2, out_channels), kernel_size=3, padding=1)
        # TODO: Examine the feature refiner design

    def forward(self, feat_fused, feat_seg):
        B = feat_fused.shape[0]

        feat_fused, H, W = self.patch_embed(feat_fused)
        feat_fused = self.conv_1(feat_fused, H, W)
        feat_fused = self.norm(feat_fused.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        feat_fused = self.conv_2(feat_fused)  # self.dff(self.conv_2(feat))

        feat_seg = self.extractor_seg(self.up(feat_seg))
        feat_seg = self.trans_1(feat_seg)

        feat_meta = self.fusor(torch.cat([feat_fused, feat_seg], dim=1))
        feat_meta = self.trans_2(feat_meta)

        return feat_meta


class FeatureTransform(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transform = DetailFeatureExtraction(in_channels)
        self.downsampler = Downsampler(in_channels)
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=1, padding='same')

    def forward(self, feat):
        feat = self.transform(feat)
        feat = self.conv(feat)
        feat = self.downsampler(feat)

        return feat


# =============================================================================

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = ChannelWiseLayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = ChannelWiseLayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    """
    Overlapped image patch embedding with 3x3 Conv
    """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    """
    Restormer encoder
    """

    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 num_blocks=(4, 4),
                 heads=(8, 8),
                 ffn_expansion_factor=2,
                 bias=False,
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias) for i in range(num_blocks[0])])

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        return out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, feat, aux_feat=None):
        if aux_feat is not None:
            out_enc_level0 = torch.cat((feat, aux_feat), dim=1)
            out_enc_level0 = self.reduce_channel(out_enc_level0)
        else:
            out_enc_level0 = feat

        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


class ZeroConv(nn.Conv2d):

    def __init__(self, channels):
        super().__init__(channels, channels, 1)

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()
            if self.bias is not None:
                self.bias.zero_()


class ControlledModule(nn.Module):
    def __init__(self, channels, module):
        super().__init__()
        self.conv_input = ZeroConv(channels)
        self.conv_output = ZeroConv(channels)
        self.module = module

    def forward(self, input, aux_input=None):
        if aux_input is not None:
            input += self.conv_input(aux_input)

        return self.conv_output(self.module(input))


class WrappedSegmentor(nn.Module):
    """
    Wrapped segmentor with image normalization and exclude its params from state dict.
    """

    def __init__(self, segmentor: BaseSegmentor, mean, std):
        super().__init__()
        self.segmentor = segmentor
        self.mean = mean
        self.std = std

    def forward(self, img, *args, **kwargs):
        img = TVF.normalize(img, self.mean, self.std, True)
        return self.segmentor(img, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        pass

    def state_dict(self, *args, **kwargs):
        return {}


# Efficient Channel Attention for Deep Convolutional Neural Networks

class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECA_NS(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA_NS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x
