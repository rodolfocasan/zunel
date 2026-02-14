# zunel/attention.py
import math
import torch
from torch import nn
from torch.nn import functional as F

import logging
from zunel import helpers





logger = logging.getLogger(__name__)

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

@torch.jit.script
def _gate(a, b, n_ch):
    n = n_ch[0]
    s = a + b
    return torch.tanh(s[:, :n, :]) * torch.sigmoid(s[:, n:, :])





class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size = 1,
        p_dropout = 0.0,
        window_size = 4,
        isflow = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.cond_layer_idx = self.n_layers

        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                self.cond_layer_idx = kwargs.get("cond_layer_idx", 2)
                assert self.cond_layer_idx < self.n_layers, "cond_layer_idx should be less than n_layers"

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttn(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size)
            )
            self.norm_layers_1.append(ChannelNorm(hidden_channels))
            self.ffn_layers.append(
                FeedForward(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(ChannelNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        
        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2)).transpose(1, 2)
                x = (x + g) * x_mask
            
            y = self.drop(self.attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_1[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)
        return x * x_mask





class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size = 1,
        p_dropout = 0.0,
        proximal_bias = False,
        proximal_init = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttn(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout = p_dropout,
                    proximal_bias = proximal_bias,
                    proximal_init = proximal_init
                )
            )
            self.norm_layers_0.append(ChannelNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttn(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
            )
            self.norm_layers_1.append(ChannelNorm(hidden_channels))
            self.ffn_layers.append(
                FeedForward(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout = p_dropout,
                    causal = True
                )
            )
            self.norm_layers_2.append(ChannelNorm(hidden_channels))

    def forward(self, x, x_mask, h, h_mask):
        self_attn_mask = helpers.causal_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        encdec_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        
        for i in range(self.n_layers):
            y = self.drop(self.self_attn_layers[i](x, x, self_attn_mask))
            x = self.norm_layers_0[i](x + y)
            y = self.drop(self.encdec_attn_layers[i](x, h, encdec_mask))
            x = self.norm_layers_1[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)
        return x * x_mask





class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout = 0.0,
        window_size = None,
        heads_share = True,
        block_length = None,
        proximal_bias = False,
        proximal_init = False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None
        self.k_channels = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_std = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_std
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_std
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self._attend(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def _attend(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_rel = self._get_rel_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_rel_keys(query / math.sqrt(self.k_channels), key_rel)
            scores = scores + self._rel_to_abs(rel_logits)

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._proximal_bias(t_s).to(device=scores.device, dtype=scores.dtype)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)

        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            rel_weights = self._abs_to_rel(p_attn)
            val_rel = self._get_rel_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_rel_vals(rel_weights, val_rel)

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_rel_vals(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_rel_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_rel_embeddings(self, rel_emb, length):
        2 * self.window_size + 1
        pad_len = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        
        if pad_len > 0:
            padded = F.pad(rel_emb, helpers.flatten_pad_spec([[0, 0], [pad_len, pad_len], [0, 0]]))
        else:
            padded = rel_emb
        return padded[:, slice_start:slice_end]

    def _rel_to_abs(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, helpers.flatten_pad_spec([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, helpers.flatten_pad_spec([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]

    def _abs_to_rel(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, helpers.flatten_pad_spec([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = F.pad(x_flat, helpers.flatten_pad_spec([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    def _proximal_bias(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)





class FeedForward(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.padding = self._causal_pad if causal else self._same_pad
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        x = x * torch.sigmoid(1.702 * x) if self.activation == "gelu" else torch.relu(x)
        x = self.conv_2(self.padding(self.drop(x) * x_mask))
        return x * x_mask

    def _causal_pad(self, x):
        if self.kernel_size == 1:
            return x
        return F.pad(x, helpers.flatten_pad_spec([[0, 0], [0, 0], [self.kernel_size - 1, 0]]))

    def _same_pad(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return F.pad(x, helpers.flatten_pad_spec([[0, 0], [0, 0], [pad_l, pad_r]]))