# zunel/core/layers.py
import math
import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from zunel.core import helpers
from zunel.core.attention import TransformerEncoder
from zunel.core.helpers import initialize_weights, compute_padding
from zunel.audio.signal_processing import rational_quadratic_transform





LRELU_SLOPE = 0.1


class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)





class DepthSepConv(nn.Module):
    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.drop = nn.Dropout(p_dropout)

        self.dw_convs = nn.ModuleList()
        self.pw_convs = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()

        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.dw_convs.append(
                nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding)
            )
            self.pw_convs.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(ChannelNorm(channels))
            self.norms_2.append(ChannelNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g

        for i in range(self.n_layers):
            y = F.gelu(self.norms_1[i](self.dw_convs[i](x * x_mask)))
            y = self.drop(F.gelu(self.norms_2[i](self.pw_convs[i](y))))
            x = x + y
        return x * x_mask





class WaveNetBlock(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = torch.nn.utils.weight_norm(
                torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight"
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.utils.weight_norm(
                torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding),
                name = "weight",
            )
            self.in_layers.append(in_layer)
            res_ch = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                torch.nn.utils.weight_norm(torch.nn.Conv1d(hidden_channels, res_ch, 1), name="weight")
            )

    def forward(self, x, x_mask, g=None, **kwargs):
        out = torch.zeros_like(x)
        n_ch_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            cond = self.cond_layer(g)
        else:
            cond = None

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if cond is not None:
                start = i * 2 * self.hidden_channels
                end = (i + 1) * 2 * self.hidden_channels
                g_l = cond[:, start:end, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = self.drop(helpers.gated_activation(x_in, g_l, n_ch_tensor))
            rs = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                x = (x + rs[:, :self.hidden_channels, :]) * x_mask
                out = out + rs[:, self.hidden_channels:, :]
            else:
                out = out + rs
        return out * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)





class ResidualStack1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=compute_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=compute_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=compute_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(initialize_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=compute_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=compute_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=compute_padding(kernel_size, 1))),
        ])
        self.convs2.apply(initialize_weights)

    def forward(self, x, x_mask=None):
        i = 0
        n = len(self.convs1)

        while i < n:
            c1 = self.convs1[i]
            c2 = self.convs2[i]

            xt = F.leaky_relu(x, LRELU_SLOPE)

            if x_mask is not None:
                xt = xt * x_mask

            xt1 = c1(xt)
            xt1 = F.leaky_relu(xt1, LRELU_SLOPE)

            if x_mask is not None:
                xt1 = xt1 * x_mask
            else:
                xt1 = xt1 * 1

            xt2 = c2(xt1)
            x = xt2 + x
            i = i + 1
        if x_mask is not None:
            return x * x_mask
        else:
            return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)





class ResidualStack2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=compute_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=compute_padding(kernel_size, dilation[1]))),
        ])
        self.convs.apply(initialize_weights)

    def forward(self, x, x_mask=None):
        i = 0
        n = len(self.convs)

        while i < n:
            c = self.convs[i]
            xt = F.leaky_relu(x, LRELU_SLOPE)

            if x_mask is not None:
                xt = xt * x_mask
            else:
                xt = xt * 1

            xt = c(xt)
            x = xt + x
            i = i + 1
        if x_mask is not None:
            return x * x_mask
        else:
            return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)





class LogLayer(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            return y, torch.sum(-y, [1, 2])
        return torch.exp(x) * x_mask





class FlipLayer(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x





class AffineLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = (self.m + torch.exp(self.logs) * x) * x_mask
            return y, torch.sum(self.logs * x_mask, [1, 2])
        return (x - self.m) * torch.exp(-self.logs) * x_mask





class CouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNetBlock(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.enc(self.pre(x0) * x_mask, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m, logs = stats, torch.zeros_like(stats)

        if not reverse:
            x1 = (m + x1 * torch.exp(logs)) * x_mask
            return torch.cat([x0, x1], 1), torch.sum(logs, [1, 2])
        else:
            return torch.cat([x0, (x1 - m) * torch.exp(-logs) * x_mask], 1)





class SplineFlow(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DepthSepConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.proj(self.convs(self.pre(x0), x_mask, g=g)) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        uw = h[..., :self.num_bins] / math.sqrt(self.proj.in_channels)
        uh = h[..., self.num_bins:2 * self.num_bins] / math.sqrt(self.proj.in_channels)
        ud = h[..., 2 * self.num_bins:]

        x1, logdet = rational_quadratic_transform(
            x1, uw, uh, ud, inverse=reverse, tails="linear", tail_bound=self.tail_bound
        )
        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logdet * x_mask, [1, 2])
        return (x, logdet) if not reverse else x