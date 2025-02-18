from functools import partial

import torch
from timm.layers import DropPath
from torch import nn

from torch.nn import functional as F, InstanceNorm2d



class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.ReLU(inplace=True),
                 pool=None, norm=nn.BatchNorm2d, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_module('conv2d', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            padding=padding, stride=stride, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
        if activation is not None:
            self.add_module('activation', activation)
        if pool is not None:
            self.add_module('pool2d', pool)


class ConvBlock(nn.Sequential):
    def __init__(self, channels, kernel_size, padding, activation=nn.ReLU(inplace=True), norm=nn.BatchNorm2d, dilation=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_module('norm', norm(channels[0]))
        self.add_module('activation', activation)
        self.add_module('conv2d', nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size,
                                            padding=padding, dilation=dilation))


class TransBlock(nn.Sequential):
    def __init__(self, channels, kernel_size, padding=0, activation=nn.ReLU, norm=nn.BatchNorm2d, pool=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.add_module('norm', norm(channels[0]))
        self.add_module('activation', activation())
        self.add_module('conv2d', nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, padding=padding))
        if pool is not None:  # TODO: Examine this pooling layer
            self.add_module('pool2d', pool(kernel_size=2, stride=2))
            self.add_module('upsample', nn.UpsamplingNearest2d(scale_factor=2))


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, growth_rate, dilation=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(num_convs):
            layers.append(ConvBlock((growth_rate * i + in_channels, growth_rate), 3, padding='same', dilation=dilation))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.net:
            y = layer(x)
            x = torch.cat((x, y), dim=1)
        return x


class RDB(nn.Sequential):
    def __init__(self, num_convs, in_channels, growth_rate):
        super().__init__()
        self.dense_block = DenseBlock(num_convs, in_channels, growth_rate)
        self.trans_block = TransBlock((num_convs * growth_rate + in_channels, in_channels), 3, 1)

    def forward(self, x):
        out = self.dense_block(x)
        out = self.trans_block(out)
        return out + x


class DRDB(nn.Sequential):
    """
    Dilated dense residual block from "Attention-guided Network for Ghost-free High Dynamic Range Imaging"
    """

    def __init__(self, num_convs, in_channels, growth_rate):
        super().__init__()
        self.dense_block = DenseBlock(num_convs, in_channels, growth_rate, dilation=2)
        self.trans_block = TransBlock((num_convs * growth_rate + in_channels, in_channels), 3, 1, pool=None)

    def forward(self, x):
        out = self.dense_block(x)
        out = self.trans_block(out)
        return out + x



# Implementation of GRN and ConvNeXt block adopted from mmpretrain
class GRN(nn.Module):
    """Global Response Normalization Module.

    Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders <http://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
                1, -1, 1, 1) + x
        return x


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
                             f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, kernel_size=7, padding='same')

        self.linear_pw_conv = linear_pw_conv
        self.norm = LayerNorm2d(in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = nn.GELU()
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.grn = GRN(mid_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        shortcut = x
        x = self.depthwise_conv(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x, data_format='channel_last')
            x = self.pointwise_conv1(x)
            x = self.act(x)
            x = self.grn(x, data_format='channel_last')
            x = self.pointwise_conv2(x)
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        else:
            x = self.norm(x, data_format='channel_first')
            x = self.pointwise_conv1(x)
            x = self.act(x)
            x = self.grn(x, data_format='channel_first')
            x = self.pointwise_conv2(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)
        return x


class Downsampler(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelWiseLayerNorm(channels)
        self.down = nn.Conv2d(channels, channels, 2, 2)

    def forward(self, input):
        return self.down(self.norm(input))


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        input = input.permute(0, 2, 3, 1)
        super().forward(input)
        return input.permute(0, 3, 1, 2)


class SPADE(nn.Module):
    """
    Spatially-adaptive denormalization (SPADE)
    https://arxiv.org/abs/1903.07291
    """

    def __init__(self, norm_nc, label_nc, nhidden=128):

        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        #actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class ResnetBlockSpade(nn.Module):
    """
    ResNet with SPADE from Context-Aware Image Inpainting with Learned Semantic Priors
    https://arxiv.org/abs/2106.07220
    """

    def __init__(self, dim, layout_dim, dilation, use_spectral_norm=True):
        super(ResnetBlockSpade, self).__init__()
        self.conv_block = nn.Sequential(
            SPADE(dim, layout_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),

            SPADE(256, layout_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),

        )

    def forward(self, x, layout):
        # out = x + self.conv_block(x)
        out = x
        for i in range(len(self.conv_block)):
            sub_block = self.conv_block[i]
            if i == 0 or i == 4:
                out = sub_block(out, layout)
            else:
                out = sub_block(out)

        out_final = out + x
        # skimage.io.imsave('block.png', out[0].detach().permute(1,2,0).cpu().numpy()[:,:,0])

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out_final


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
