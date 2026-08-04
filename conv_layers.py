"""
conv_layers.py
================
Optional conv-layer replacements for the generator/discriminator's `ConvLayer` ablation
knob (see build_model.py). Every class here has the SAME drop-in signature as the
original `PlainConv2d` / `DeformableConv2d`:

    ConvLayer(in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False)
    forward(x: (B, in_channels, H, W)) -> (B, out_channels, H, W)

so any of them can be selected via `--conv_type` (see config.py) without touching
build_model.py's InverseModel / Discriminator definitions.

Sources (numerically faithful to, or -- where noted -- lightly adapted from):
  - DCNv4Conv    : official CUDA op, OpenGVLab/DCNv4 (CVPR'24)      -- optional dependency
  - InvolutionConv: Li et al., "Involution", CVPR 2021 (d-li14/involution, involution_naive.py)
  - DynamicConv  : Chen et al., "Dynamic Convolution", CVPR 2020 (community ref: kaijieshi7/Dynamic-convolution-Pytorch)
  - ODConvLayer  : Li et al., "Omni-Dimensional Dynamic Convolution", ICLR 2022 (OSVAI/ODConv, modules/odconv.py)

Adaptation note (DCNv4Conv / InvolutionConv only): both official ops are channel-PRESERVING
(single `channels` arg -- no separate in/out). Since every conv in this codebase changes
channel count, we prepend a 1x1 nn.Conv2d projection (in_channels -> out_channels) and then
apply the op at that width. Both ops also require out_channels to be a multiple of 16
(DCNv4's CUDA kernel asserts `channels // group % 16 == 0`; Involution hardcodes
group_channels=16). The only layer in this codebase that violates this is the generator's
final 1-channel output conv -- for that one layer we fall back to DeformableConv2d and log
a warning. DynamicConv and ODConvLayer need no projection/fallback: their kernel-mixture
attention works for arbitrary in/out channel combinations natively.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from Deform_convII import DeformableConv2d

try:
    from DCNv4.modules.dcnv4 import DCNv4 as _OfficialDCNv4
except ImportError:
    try:
        from DCNv4.modules import DCNv4 as _OfficialDCNv4
    except ImportError:
        _OfficialDCNv4 = None


def _to_square_int(kernel_size):
    if isinstance(kernel_size, (tuple, list)):
        if kernel_size[0] != kernel_size[1]:
            raise ValueError(f"{__name__} conv layers require a square kernel_size, got {kernel_size!r}.")
        return kernel_size[0]
    return kernel_size


def _to_int(stride_or_dilation):
    if isinstance(stride_or_dilation, (tuple, list)):
        return stride_or_dilation[0]
    return stride_or_dilation


def _same_padding(kernel_size):
    return (kernel_size - 1) // 2


# --------------------------------------------------------------------------- #
# Plain nn.Conv2d, same signature as everything else -- moved here from
# build_model.py so all ConvLayer options live in one place.
# --------------------------------------------------------------------------- #
class PlainConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False):
        super().__init__()
        kernel_size = _to_square_int(kernel_size) if not isinstance(kernel_size, int) else kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if padding == 'same':
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------------------------------- #
# DCNv4 -- official CUDA op (channel-preserving), wrapped with a 1x1 projection.
# --------------------------------------------------------------------------- #
class DCNv4Conv(nn.Module):
    """Requires: pip install --index-url https://test.pypi.org/simple/ dcnv4
    (GPU + matching CUDA build; see https://github.com/OpenGVLab/DCNv4)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False):
        super().__init__()
        kernel_size = _to_square_int(kernel_size)
        stride = _to_int(stride)
        if padding == 'same':
            padding = _same_padding(kernel_size)

        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.out_channels = out_channels

        if out_channels % 16 != 0:
            warnings.warn(
                f"DCNv4Conv: out_channels={out_channels} is not a multiple of 16, but the "
                f"official DCNv4 CUDA kernel requires channels // group to be a multiple of "
                f"16. Falling back to DeformableConv2d for this layer only."
            )
            self.dcn = None
            self.fallback = DeformableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
            return
        self.fallback = None

        if _OfficialDCNv4 is None:
            raise ImportError(
                "--conv_type dcnv4 requires the official OpenGVLab DCNv4 package, which is not "
                "installed. Install it with:\n"
                "  pip install --index-url https://test.pypi.org/simple/ dcnv4\n"
                "(requires a GPU + CUDA build matching your torch install; "
                "see https://github.com/OpenGVLab/DCNv4)."
            )
        group = out_channels // 16  # smallest group satisfying `channels // group % 16 == 0`
        self.dcn = _OfficialDCNv4(
            channels=out_channels, kernel_size=kernel_size, stride=stride,
            pad=padding, group=group,
        )

    def forward(self, x):
        x = self.project(x)
        if self.fallback is not None:
            return self.fallback(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # NCHW -> (N, L, C), as DCNv4 expects
        x = self.dcn(x, shape=(h, w))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x


# --------------------------------------------------------------------------- #
# Involution -- content-adaptive, channel-agnostic-within-group spatial kernels.
# Faithful to d-li14/involution's involution_naive.py (reduction_ratio=4,
# group_channels=16 fixed), wrapped with a 1x1 projection.
# --------------------------------------------------------------------------- #
class InvolutionConv(nn.Module):
    """Involution's kernel-generation is channel-preserving with a FIXED group_channels=16,
    so out_channels must be a multiple of 16. Rather than falling back to a different op for
    channel counts that don't satisfy this (e.g. this codebase's 1-channel generator output),
    we project up to the nearest multiple of 16 (`mid_channels`, min 16), run real involution
    there, then 1x1-project back down to out_channels if they differ. So involution always
    runs -- there's no silent substitution for any layer."""
    _REDUCTION_RATIO = 4
    _GROUP_CHANNELS = 16

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False):
        super().__init__()
        kernel_size = _to_square_int(kernel_size)
        stride = _to_int(stride)

        mid_channels = max(self._GROUP_CHANNELS,
                            -(-out_channels // self._GROUP_CHANNELS) * self._GROUP_CHANNELS)  # ceil to multiple of 16

        self.project_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=bias)
        self.project_out = (nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=bias)
                             if mid_channels != out_channels else None)
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.groups = mid_channels // self._GROUP_CHANNELS
        reduced = max(mid_channels // self._REDUCTION_RATIO, 1)
        self.reduce = nn.Sequential(
            nn.Conv2d(mid_channels, reduced, 1),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )
        self.span = nn.Conv2d(reduced, kernel_size ** 2 * self.groups, 1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=(kernel_size - 1) // 2, stride=stride)

    def forward(self, x):
        x = self.project_in(x)
        weight = self.span(self.reduce(x if self.stride == 1 else self.avgpool(x)))
        b, _, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self._GROUP_CHANNELS, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.mid_channels, h, w)
        if self.project_out is not None:
            out = self.project_out(out)
        return out


# --------------------------------------------------------------------------- #
# Dynamic Convolution -- Chen et al., CVPR 2020. K parallel kernels combined via
# per-sample softmax attention (annealed temperature 34 -> 1, step 3 -- matches
# the paper's schedule). Works for any in/out channels; no projection needed.
# --------------------------------------------------------------------------- #
class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False,
                 K=4, ratio=0.25, temperature=34):
        super().__init__()
        kernel_size = _to_square_int(kernel_size)
        stride = _to_int(stride)
        if padding == 'same':
            padding = _same_padding(kernel_size)
        assert temperature % 3 == 1, "temperature must satisfy temperature % 3 == 1 (annealing step is 3, floor is 1)"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.K = K
        self.temperature = temperature

        hidden = K if in_channels == 3 else max(int(in_channels * ratio), 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden, K, 1, bias=True)

        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(K, out_channels)) if bias else None

        for i in range(K):
            nn.init.kaiming_uniform_(self.weight[i])
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)

    def step_temperature(self):
        """Anneal 34 -> 31 -> ... -> 1, called once per training epoch from train_fn.py."""
        if self.temperature != 1:
            self.temperature = max(1, self.temperature - 3)

    def forward(self, x):
        attn = self.avgpool(x)
        attn = F.relu(self.fc1(attn))
        attn = self.fc2(attn).view(x.size(0), -1)
        attn = F.softmax(attn / self.temperature, dim=1)

        b, c, h, w = x.shape
        x = x.reshape(1, -1, h, w)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(attn, weight).view(
            b * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        aggregate_bias = torch.mm(attn, self.conv_bias).view(-1) if self.conv_bias is not None else None

        out = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias,
                        stride=self.stride, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, out.size(-2), out.size(-1))
        return out


# --------------------------------------------------------------------------- #
# ODConv -- Li et al., ICLR 2022. 4D kernel attention (spatial / in-channel /
# out-channel / kernel-index). Faithful port of OSVAI/ODConv's modules/odconv.py
# (Attention + ODConv2d), adapted to this project's constructor signature.
# Works for any in/out channels; no projection needed.
# --------------------------------------------------------------------------- #
class _ODAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        return torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_filter_attention(self, x):
        return torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        return torch.sigmoid(spatial_attention / self.temperature)

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        return F.softmax(kernel_attention / self.temperature, dim=1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=False,
                 reduction=0.0625, kernel_num=4):
        super().__init__()
        kernel_size = _to_square_int(kernel_size)
        stride = _to_int(stride)
        if padding == 'same':
            padding = _same_padding(kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_num = kernel_num

        self.attention = _ODAttention(in_channels, out_channels, kernel_size, groups=1,
                                       reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_channels, in_channels, kernel_size, kernel_size))
        for i in range(kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')
        self.conv_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self._pointwise = (kernel_size == 1 and kernel_num == 1)

    def _forward_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        b, c, h, w = x.shape
        x = x * channel_attention
        x = x.reshape(1, -1, h, w)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_channels, self.kernel_size, self.kernel_size])
        out = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride,
                        padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, out.size(-2), out.size(-1))
        return out * filter_attention

    def _forward_pw1x(self, x):
        channel_attention, filter_attention, _, _ = self.attention(x)
        x = x * channel_attention
        out = F.conv2d(x, weight=self.weight.squeeze(0), bias=None, stride=self.stride, padding=self.padding)
        return out * filter_attention

    def forward(self, x):
        out = self._forward_pw1x(x) if self._pointwise else self._forward_common(x)
        if self.conv_bias is not None:
            out = out + self.conv_bias.view(1, -1, 1, 1)
        return out


CONV_LAYER_REGISTRY = {
    "plain": PlainConv2d,
    "deform": DeformableConv2d,
    "dcnv4": DCNv4Conv,
    "involution": InvolutionConv,
    "dynamic": DynamicConv,
    "odconv": ODConvLayer,
}
