# zunel/signal_processing.py
import librosa
import numpy as np
from librosa.filters import mel as librosa_mel_fn

import torch
import torch.utils.data
from torch.nn import functional as F


MAX_WAV_VALUE = 32768.0

_mel_basis_cache = {}
_window_cache = {}

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def compress_dynamic_range(x, factor=1, floor=1e-5):
    return torch.log(torch.clamp(x, min=floor) * factor)

def decompress_dynamic_range(x, factor=1):
    return torch.exp(x) / factor

def normalize_spectrum(magnitudes):
    return compress_dynamic_range(magnitudes)

def denormalize_spectrum(magnitudes):
    return decompress_dynamic_range(magnitudes)


def compute_spectrogram(y, n_fft, sample_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.1:
        print('[zunel] min value is ', torch.min(y))
    if torch.max(y) > 1.1:
        print('[zunel] max value is ', torch.max(y))

    global _window_cache
    cache_key = str(win_size) + '_' + str(y.dtype) + '_' + str(y.device)
    if cache_key not in _window_cache:
        _window_cache[cache_key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect',
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_window_cache[cache_key],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return spec.abs()


def spec_to_mel(spec, n_fft, n_mels, sample_rate, fmin, fmax):
    global _mel_basis_cache
    cache_key = str(fmax) + '_' + str(spec.dtype) + '_' + str(spec.device)

    if cache_key not in _mel_basis_cache:
        mel_fb = librosa_mel_fn(sample_rate, n_fft, n_mels, fmin, fmax)
        _mel_basis_cache[cache_key] = torch.from_numpy(mel_fb).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(_mel_basis_cache[cache_key], spec)
    return normalize_spectrum(spec)


def compute_mel_spectrogram(y, n_fft, n_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print('[zunel] min value is ', torch.min(y))
    if torch.max(y) > 1.0:
        print('[zunel] max value is ', torch.max(y))

    global _mel_basis_cache, _window_cache
    mel_key = str(fmax) + '_' + str(y.dtype) + '_' + str(y.device)
    win_key = str(win_size) + '_' + str(y.dtype) + '_' + str(y.device)

    if mel_key not in _mel_basis_cache:
        mel_fb = librosa_mel_fn(sample_rate, n_fft, n_mels, fmin, fmax)
        _mel_basis_cache[mel_key] = torch.from_numpy(mel_fb).to(dtype=y.dtype, device=y.device)
    if win_key not in _window_cache:
        _window_cache[win_key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect',
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=_window_cache[win_key], center=center, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True,
    )
    spec = spec.abs()
    spec = torch.matmul(_mel_basis_cache[mel_key], spec)
    return normalize_spectrum(spec)


def rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if tails is None:
        spline_fn = _bounded_rq_spline
        spline_kwargs = {}
    else:
        spline_fn = _unbounded_rq_spline
        spline_kwargs = {'tails': tails, 'tail_bound': tail_bound}

    return spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs,
    )


def _bin_search(bin_locs, inputs, eps=1e-6):
    bin_locs[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locs, dim=-1) - 1


def _unbounded_rq_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails='linear',
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside = ~inside

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant
        outputs[outside] = inputs[outside]
        logabsdet[outside] = 0
    else:
        raise RuntimeError('[zunel] {} tails are not implemented.'.format(tails))

    outputs[inside], logabsdet[inside] = _bounded_rq_spline(
        inputs=inputs[inside],
        unnormalized_widths=unnormalized_widths[inside, :],
        unnormalized_heights=unnormalized_heights[inside, :],
        unnormalized_derivatives=unnormalized_derivatives[inside, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return outputs, logabsdet


def _bounded_rq_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError('[zunel] Input to a transform is not within its domain')

    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError('[zunel] Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('[zunel] Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = (_bin_search(cumheights if inverse else cumwidths, inputs))[..., None]

    w0 = cumwidths.gather(-1, bin_idx)[..., 0]
    bw = widths.gather(-1, bin_idx)[..., 0]
    h0 = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    bd = delta.gather(-1, bin_idx)[..., 0]
    d_k = derivatives.gather(-1, bin_idx)[..., 0]
    d_k1 = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    bh = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - h0) * (d_k + d_k1 - 2 * bd) + bh * (bd - d_k)
        b = bh * d_k - (inputs - h0) * (d_k + d_k1 - 2 * bd)
        c = -bd * (inputs - h0)

        disc = b.pow(2) - 4 * a * c
        assert (disc >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(disc))
        outputs = root * bw + w0

        t1mt = root * (1 - root)
        denom = bd + ((d_k + d_k1 - 2 * bd) * t1mt)
        numer = bd.pow(2) * (d_k1 * root.pow(2) + 2 * bd * t1mt + d_k * (1 - root).pow(2))
        logabsdet = torch.log(numer) - 2 * torch.log(denom)
        return outputs, -logabsdet
    else:
        theta = (inputs - w0) / bw
        t1mt = theta * (1 - theta)

        numer = bh * (bd * theta.pow(2) + d_k * t1mt)
        denom = bd + ((d_k + d_k1 - 2 * bd) * t1mt)
        outputs = h0 + numer / denom

        d_numer = bd.pow(2) * (d_k1 * theta.pow(2) + 2 * bd * t1mt + d_k * (1 - theta).pow(2))
        logabsdet = torch.log(d_numer) - 2 * torch.log(denom)
        return outputs, logabsdet