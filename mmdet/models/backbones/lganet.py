# LGANet backbone for MMDetection
# Ported from LS-DETR (ultralytics/nn/extra_modules/block.py,
# ultralytics/nn/extra_modules/attention.py, ultralytics/nn/modules/block.py,
# ultralytics/nn/modules/conv.py)
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch.nn import BatchNorm2d

from mmdet.registry import MODELS

# ---------------------------------------------------------------------------
# timm DropPath
# ---------------------------------------------------------------------------
try:
    from timm.models.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample."""
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor = torch.floor(random_tensor + keep_prob)
            return x / keep_prob * random_tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def get_activation(act: str, inplace: bool = True):
    act = act.lower()
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError(f'Unknown activation: {act}')
    if hasattr(m, 'inplace'):
        m.inplace = inplace
    return m


# ---------------------------------------------------------------------------
# Conv (from ultralytics/nn/modules/conv.py)
# ---------------------------------------------------------------------------
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# ---------------------------------------------------------------------------
# RepConv (from ultralytics/nn/modules/conv.py)
# ---------------------------------------------------------------------------
class RepConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
            return 0, 0
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.forward = self.forward_fuse


# ---------------------------------------------------------------------------
# ConvNormLayer (from ultralytics/nn/modules/block.py)
# ---------------------------------------------------------------------------
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# ---------------------------------------------------------------------------
# BasicBlock (from ultralytics/nn/modules/block.py)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        short = x if self.shortcut else self.short(x)
        out = out + short
        out = self.act(out)
        return out


# ---------------------------------------------------------------------------
# EMA attention (from ultralytics/nn/extra_modules/attention.py)
# ---------------------------------------------------------------------------
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# ConvolutionalGLU (from ultralytics/nn/extra_modules/block.py)
# ---------------------------------------------------------------------------
class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


# ---------------------------------------------------------------------------
# Partial_conv3 / Partial_conv3_Rep
# ---------------------------------------------------------------------------
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class Partial_conv3_Rep(Partial_conv3):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__(dim, n_div, forward)
        self.partial_conv3 = RepConv(self.dim_conv3, self.dim_conv3, k=3, act=False, bn=False)


# ---------------------------------------------------------------------------
# Faster_Block_CGLU / Faster_Block_EMA_CGLU / Faster_Block_Rep_EMA_CGLU
# ---------------------------------------------------------------------------
class Faster_Block_CGLU(nn.Module):
    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1,
                 layer_scale_init_value=0.0, pconv_fw_type='split_cat'):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        self.mlp = ConvolutionalGLU(dim)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.forward = self.forward_layer_scale

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        return shortcut + self.drop_path(self.mlp(x))

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        return shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))


class Faster_Block_EMA_CGLU(Faster_Block_CGLU):
    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1,
                 layer_scale_init_value=0, pconv_fw_type='split_cat'):
        super().__init__(inc, dim, n_div, mlp_ratio, drop_path, layer_scale_init_value, pconv_fw_type)
        self.attention = EMA(channels=dim)

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        return shortcut + self.attention(self.drop_path(self.mlp(x)))

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        return shortcut + self.attention(
            self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)))


class Faster_Block_Rep_EMA_CGLU(Faster_Block_EMA_CGLU):
    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1,
                 layer_scale_init_value=0, pconv_fw_type='split_cat'):
        super().__init__(inc, dim, n_div, mlp_ratio, drop_path, layer_scale_init_value, pconv_fw_type)
        self.spatial_mixing = Partial_conv3_Rep(dim, n_div, pconv_fw_type)


# ---------------------------------------------------------------------------
# BasicBlock_Faster_Block_Rep_EMA_CGLU
# ---------------------------------------------------------------------------
class BasicBlock_Faster_Block_Rep_EMA_CGLU(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.branch2b = Faster_Block_Rep_EMA_CGLU(ch_out, ch_out)


# ---------------------------------------------------------------------------
# Blocks container (from ultralytics/nn/modules/block.py)
# ---------------------------------------------------------------------------
class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', variant='d'):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act,
                )
            )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# LGANet backbone
# ---------------------------------------------------------------------------
@MODELS.register_module()
class LGANet(BaseModule):
    """LGANet backbone ported from LS-DETR.

    Architecture:
        stem  : 3x ConvNormLayer + MaxPool2d  -> /4
        layer1: 2x BasicBlock_Faster_Block_Rep_EMA_CGLU, 64ch   (P2/4)
        layer2: 2x BasicBlock_Faster_Block_Rep_EMA_CGLU, 128ch  (P3/8)
        layer3: 2x BasicBlock_Faster_Block_Rep_EMA_CGLU, 256ch  (P4/16)
        layer4: 2x BasicBlock_Faster_Block_Rep_EMA_CGLU, 512ch  (P5/32)

    out_indices: 0-based index into [layer1, layer2, layer3, layer4].
        e.g. out_indices=(1, 2, 3) returns (P3, P4, P5).

    Output channels per stage: [64, 128, 256, 512].
    """

    # output channels for each of the 4 stages
    stage_channels = (64, 128, 256, 512)

    def __init__(self,
                 out_indices=(1, 2, 3),
                 frozen_stages=-1,
                 norm_eval=False,
                 act='relu',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert max(out_indices) <= 3, 'out_indices must be in range [0, 3]'
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # stem: /4 downsampling
        self.stem = nn.Sequential(
            ConvNormLayer(3, 32, 3, 2, act=act),
            ConvNormLayer(32, 32, 3, 1, act=act),
            ConvNormLayer(32, 64, 3, 1, act=act),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 stages
        self.layer1 = Blocks(64,  64,  BasicBlock_Faster_Block_Rep_EMA_CGLU, 2, stage_num=2, act=act)
        self.layer2 = Blocks(64,  128, BasicBlock_Faster_Block_Rep_EMA_CGLU, 2, stage_num=3, act=act)
        self.layer3 = Blocks(128, 256, BasicBlock_Faster_Block_Rep_EMA_CGLU, 2, stage_num=4, act=act)
        self.layer4 = Blocks(256, 512, BasicBlock_Faster_Block_Rep_EMA_CGLU, 2, stage_num=5, act=act)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= -1:
            # freeze stem when frozen_stages >= 0
            if self.frozen_stages >= 0:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, BatchNorm2d):
                    m.eval()
