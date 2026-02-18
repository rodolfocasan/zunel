# zunel/processing.py
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import medfilt2d, medfilt, stft, istft, hann





def enhance_tts(input_path, output_path, sr=22050):
    audio, _ = librosa.load(input_path, sr=sr)
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95

    sos = signal.butter(10, [80, 8000], btype='band', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)

    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    frame_length, hop_length = 2048, 512
    stft_result = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude = signal.medfilt(np.abs(stft_result), kernel_size=(1, 5))
    phase = np.angle(stft_result)
    audio = librosa.istft(magnitude * np.exp(1j * phase), hop_length=hop_length)

    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
    rms = np.sqrt(np.mean(audio**2))
    audio = audio * (0.1 / (rms + 1e-8))
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(output_path, audio, sr)


def _interpolate_spike_frames(magnitude, spike_indices):
    n_frames = magnitude.shape[1]
    spike_set = set(spike_indices.tolist())
    result = magnitude.copy()

    for i in spike_indices:
        left = i - 1
        right = i + 1

        while left in spike_set and left > 0:
            left -= 1
        while right in spike_set and right < n_frames - 1:
            right += 1

        left = max(0, left)
        right = min(n_frames - 1, right)

        if left not in spike_set and right not in spike_set:
            result[:, i] = (result[:, left] + result[:, right]) / 2.0
        elif left not in spike_set:
            result[:, i] = result[:, left]
        elif right not in spike_set:
            result[:, i] = result[:, right]
    return result


def remove_voice_conversion_artifacts(audio, sr, n_fft=1024, hop_length=256):
    original_len = len(audio)
    original_rms = np.sqrt(np.mean(audio ** 2) + 1e-8)

    n_overlap = n_fft - hop_length
    win = hann(n_fft)

    _, _, stft_matrix = stft(
        audio,
        fs = sr,
        window = win,
        nperseg = n_fft,
        noverlap = n_overlap,
        return_onesided = True
    )

    magnitude = np.abs(stft_matrix).astype(np.float32)
    phase = np.angle(stft_matrix)

    magnitude_smooth = medfilt2d(magnitude, kernel_size=(1, 3))

    frame_energy = magnitude_smooth.sum(axis=0)
    local_median = medfilt(frame_energy, kernel_size=7)
    local_median = np.where(local_median < 1e-8, 1e-8, local_median)
    spike_ratio = frame_energy / local_median

    spike_indices = np.where(spike_ratio > 3.5)[0]
    if spike_indices.size > 0:
        magnitude_smooth = _interpolate_spike_frames(magnitude_smooth, spike_indices)

    stft_smooth = magnitude_smooth * np.exp(1j * phase)

    _, audio_out = istft(
        stft_smooth,
        fs = sr,
        window = win,
        nperseg = n_fft,
        noverlap = n_overlap,
        input_onesided = True
    )

    audio_out = audio_out.astype(np.float32)

    if len(audio_out) >= original_len:
        audio_out = audio_out[:original_len]
    else:
        audio_out = np.pad(audio_out, (0, original_len - len(audio_out)))

    output_rms = np.sqrt(np.mean(audio_out ** 2) + 1e-8)
    audio_out = audio_out * (original_rms / output_rms)

    return np.clip(audio_out, -1.0, 1.0)