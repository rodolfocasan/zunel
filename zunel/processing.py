# zunel/processing.py
import librosa
import numpy as np
import soundfile as sf
from scipy import signal, ndimage





def enhance_tts(input_path, output_path, sr=22050):
    audio, _ = librosa.load(input_path, sr=sr)
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95

    sos = signal.butter(10, [80, 8000], btype='band', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)

    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    frame_length, hop_length = 2048, 512
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude = signal.medfilt(np.abs(stft), kernel_size=(1, 5))
    phase = np.angle(stft)
    audio = librosa.istft(magnitude * np.exp(1j * phase), hop_length=hop_length)

    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
    rms = np.sqrt(np.mean(audio**2))
    audio = audio * (0.1 / (rms + 1e-8))
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(output_path, audio, sr)


def _robust_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - med) / (1.4826 * mad)).astype(np.float32)


def _detect_artifact_frames(mag, flatness_thr=2.5, energy_thr=3.0, flux_thr=2.5):
    flatness = librosa.feature.spectral_flatness(S=mag + 1e-8)[0]
    frame_energy = np.mean(mag ** 2, axis=0)
    flux_raw = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux_raw])

    flat_z = _robust_zscore(flatness)
    energy_z = _robust_zscore(frame_energy)
    flux_z = _robust_zscore(flux)

    artifact = (
        (flat_z > flatness_thr) |
        (energy_z > energy_thr) |
        ((flux_z > flux_thr) & (flat_z > 1.0))
    )

    artifact = ndimage.binary_dilation(artifact, iterations=1)
    return artifact


def _repair_spectral_artifacts(mag, artifact_mask, max_interp_frames=6):
    mag_fixed = mag.copy()
    n_frames = mag.shape[1]

    labeled, n_regions = ndimage.label(artifact_mask)

    for region_id in range(1, n_regions + 1):
        indices = np.where(labeled == region_id)[0]
        start = int(indices[0])
        end = int(indices[-1]) + 1
        region_len = end - start

        left = start - 1
        right = end
        has_left = left >= 0 and not artifact_mask[left]
        has_right = right < n_frames and not artifact_mask[right]

        if region_len > max_interp_frames:
            ctx_l = max(0, start - max_interp_frames)
            ctx_r = min(n_frames, end + max_interp_frames)
            ctx = np.concatenate([
                mag[:, ctx_l:start],
                mag[:, end:ctx_r]
            ], axis=1)
            if ctx.shape[1] > 0:
                fill = np.median(ctx, axis=1, keepdims=True)
                mag_fixed[:, start:end] = fill
            continue

        if not has_left and not has_right:
            continue
        elif not has_left:
            mag_fixed[:, start:end] = mag[:, right:right + 1]
        elif not has_right:
            mag_fixed[:, start:end] = mag[:, left:left + 1]
        else:
            for i in range(region_len):
                alpha = (i + 1.0) / (region_len + 1.0)
                mag_fixed[:, start + i] = (
                    (1.0 - alpha) * mag[:, left] +
                    alpha * mag[:, right]
                )
    return mag_fixed


def repair_voice_artifacts(audio, sr, n_fft=1024, hop_size=256):
    if len(audio) < n_fft * 2:
        return audio

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_size, win_length=n_fft)
    mag = np.abs(stft)
    phase = np.angle(stft)

    artifact_mask = _detect_artifact_frames(mag)

    if not np.any(artifact_mask):
        return audio

    artifact_ratio = float(np.mean(artifact_mask))

    if artifact_ratio > 0.5:
        mag_smooth = ndimage.uniform_filter1d(mag, size=3, axis=1)
        mag_out = 0.75 * mag + 0.25 * mag_smooth
    else:
        mag_out = _repair_spectral_artifacts(mag, artifact_mask)

    stft_fixed = mag_out * np.exp(1j * phase)
    audio_fixed = librosa.istft(
        stft_fixed,
        hop_length = hop_size,
        win_length = n_fft,
        length = len(audio)
    )

    original_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    fixed_rms = np.sqrt(np.mean(audio_fixed ** 2) + 1e-10)
    if fixed_rms > 1e-10:
        audio_fixed = audio_fixed * (original_rms / fixed_rms)

    audio_fixed = np.clip(audio_fixed, -1.0, 1.0)
    return audio_fixed.astype(np.float32)