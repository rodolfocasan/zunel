# zunel/core/helpers.py
import re
import math
import json
import numpy as np

import torch
from torch.nn import functional as F





def initialize_weights(module, mean=0.0, std=0.01):
    cls = module.__class__.__name__
    if cls.find("Conv") != -1:
        module.weight.data.normal_(mean, std)


def compute_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def flatten_pad_spec(pad_spec):
    rev = pad_spec[::-1]
    
    result = []
    for pair in rev:
        for v in pair:
            result.append(v)
    return result


def interleave_with(lst, item):
    out = [item] * (len(lst) * 2 + 1)
    out[1::2] = lst
    return out


def kl_div(mean_p, log_scale_p, mean_q, log_scale_q):
    loss = (log_scale_q - log_scale_p) - 0.5
    loss += 0.5 * (torch.exp(2.0 * log_scale_p) + ((mean_p - mean_q) ** 2)) * torch.exp(-2.0 * log_scale_q)
    return loss


def sample_gumbel(shape):
    u = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(u))


def sample_gumbel_like(x):
    return sample_gumbel(x.size()).to(dtype=x.dtype, device=x.device)


def extract_segments(x, start_indices, seg_size=4):
    out = torch.zeros_like(x[:, :, :seg_size])
    for i in range(x.size(0)):
        idx = start_indices[i]
        out[i] = x[i, :, idx:idx + seg_size]
    return out


def random_extract_segments(x, x_lengths=None, seg_size=4):
    b, d, t = x.size()
    
    if x_lengths is None:
        x_lengths = t
        
    max_start = x_lengths - seg_size + 1
    starts = (torch.rand([b]).to(device=x.device) * max_start).to(dtype=torch.long)
    return extract_segments(x, starts, seg_size), starts


def sinusoidal_encoding(length, channels, min_scale=1.0, max_scale=1.0e4):
    pos = torch.arange(length, dtype=torch.float)
    n_scales = channels // 2
    log_step = math.log(float(max_scale) / float(min_scale)) / (n_scales - 1)
    inv_scales = min_scale * torch.exp(torch.arange(n_scales, dtype=torch.float) * -log_step)
    scaled = pos.unsqueeze(0) * inv_scales.unsqueeze(1)
    
    enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], 0)
    enc = F.pad(enc, [0, 0, 0, channels % 2])
    return enc.view(1, channels, length)


def add_positional_encoding(x, min_scale=1.0, max_scale=1.0e4):
    b, channels, length = x.size()
    enc = sinusoidal_encoding(length, channels, min_scale, max_scale)
    return x + enc.to(dtype=x.dtype, device=x.device)


def concat_positional_encoding(x, min_scale=1.0, max_scale=1.0e4, axis=1):
    b, channels, length = x.size()
    enc = sinusoidal_encoding(length, channels, min_scale, max_scale)
    return torch.cat([x, enc.to(dtype=x.dtype, device=x.device)], axis)


def causal_mask(length):
    return torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)


@torch.jit.script
def gated_activation(a, b, n_ch):
    n = n_ch[0]
    combined = a + b
    return torch.tanh(combined[:, :n, :]) * torch.sigmoid(combined[:, n:, :])


def shift_right(x):
    return F.pad(x, flatten_pad_spec([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]


def length_mask(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    
    idx = torch.arange(max_len, dtype=lengths.dtype, device=lengths.device)
    return idx.unsqueeze(0) < lengths.unsqueeze(1)


def build_alignment_path(durations, mask):
    b, _, t_y, t_x = mask.shape
    cum_dur = torch.cumsum(durations, -1)
    cum_flat = cum_dur.view(b * t_x)
    
    path = length_mask(cum_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, flatten_pad_spec([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def clip_gradients(params, clip_val, norm_type=2):
    if isinstance(params, torch.Tensor):
        params = [params]

    filtered_params = []
    for p in params:
        if p.grad is not None:
            filtered_params.append(p)
    
    params = filtered_params
    norm_type = float(norm_type)
    if clip_val is not None:
        clip_val = float(clip_val)

    total_norm = 0
    for p in params:
        pnorm = p.grad.data.norm(norm_type)
        total_norm += pnorm.item() ** norm_type

        if clip_val is not None:
            p.grad.data.clamp_(min=-clip_val, max=clip_val)
    return total_norm ** (1.0 / norm_type)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    return Config(**json.loads(raw))





class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = Config(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def encode_string_bits(text, pad_len=8):
    ascii_vals = []
    for c in text:
        ascii_vals.append(ord(c))

    bit_strs = []
    for v in ascii_vals:
        b = bin(v)[2:]
        b = b.zfill(8)
        bit_strs.append(b)

    bit_rows = []
    for s in bit_strs:
        row = []
        for b in s:
            row.append(int(b))
        bit_rows.append(row)

    arr = np.array(bit_rows)
    full = np.zeros((pad_len, 8), dtype=arr.dtype)
    full[:, 2] = 1
    n = min(pad_len, len(arr))
    full[:n] = arr[:n]
    return full


def decode_bits_string(bits_array):
    bin_strs = []
    for row in bits_array:
        s = ""
        for b in row:
            s = s + str(b)
        bin_strs.append(s)

    result = ""
    for s in bin_strs:
        char = chr(int(s, 2))
        result = result + char
    return result


def segment_text(text, min_len=10, language_str='[EN]'):
    if language_str in ['EN']:
        return _segment_latin(text, min_len=min_len)
    return _segment_zh(text, min_len=min_len)


def _segment_latin(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    text = re.sub('[""]', '"', text)
    text = re.sub('[\u2018\u2019]', "'", text)
    text = re.sub(r'[\<\>\(\)\[\]\"\«\»]+', '', text)
    text = re.sub('[\n\t ]+', ' ', text)
    text = re.sub('([,.!?;])', r'\1 $#!', text)

    split_parts = text.split('$#!')
    parts = []
    for s in split_parts:
        parts.append(s.strip())

    if len(parts[-1]) == 0:
        del parts[-1]

    chunks = []
    cur = []
    cur_len = 0
    for i in range(len(parts)):
        seg = parts[i]
        cur.append(seg)
        
        words = seg.split(' ')
        cur_len = cur_len + len(words)
        if cur_len > min_len or i == len(parts) - 1:
            chunk_text = ""
            for j in range(len(cur)):
                if j == 0:
                    chunk_text = cur[j]
                else:
                    chunk_text = chunk_text + " " + cur[j]
            chunks.append(chunk_text)
            cur = []
            cur_len = 0
    return _merge_short_latin(chunks)


def _merge_short_latin(segs):
    out = []
    for s in segs:
        if out and len(out[-1].split(' ')) <= 2:
            out[-1] = out[-1] + ' ' + s
        else:
            out.append(s)
    try:
        if len(out[-1].split(' ')) <= 2:
            out[-2] = out[-2] + ' ' + out[-1]
            out.pop(-1)
    except Exception:
        pass
    return out


def _segment_zh(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    text = re.sub('[\n\t ]+', ' ', text)
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    
    parts = [s.strip() for s in text.split('$#!')]
    if len(parts[-1]) == 0:
        del parts[-1]
    
    chunks, cur, cur_len = [], [], 0
    for i, seg in enumerate(parts):
        cur.append(seg)
        cur_len += len(seg)
        if cur_len > min_len or i == len(parts) - 1:
            chunks.append(' '.join(cur))
            cur, cur_len = [], 0
    return _merge_short_zh(chunks)


def _merge_short_zh(segs):
    out = []
    for s in segs:
        if out and len(out[-1]) <= 2:
            out[-1] = out[-1] + ' ' + s
        else:
            out.append(s)
    try:
        if len(out[-1]) <= 2:
            out[-2] = out[-2] + ' ' + out[-1]
            out.pop(-1)
    except Exception:
        pass
    return out