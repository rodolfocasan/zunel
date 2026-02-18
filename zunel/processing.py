# zunel/processing.py
import pyworld as pw
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d


_FRAME_PERIOD = 5.0
_F0_MEDIAN_KERNEL = 5
_F0_SIGMA = 1.0
_SP_SIGMA = 0.5


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


def remove_voice_artifacts(audio, sr):
    if len(audio) < int(sr * 0.1):
        return audio

    audio_d = np.clip(audio, -1.0, 1.0).astype(np.float64)

    ref_rms = np.sqrt(np.mean(audio_d ** 2) + 1e-12)

    f0_raw, t = pw.dio(audio_d, sr, frame_period=_FRAME_PERIOD)
    f0 = pw.stonemask(audio_d, f0_raw, t, sr)
    sp = pw.cheaptrick(audio_d, f0, t, sr)
    ap = pw.d4c(audio_d, f0, t, sr)

    voiced_idx = np.where(f0 > 0)[0]
    f0_smooth = f0.copy()

    if len(voiced_idx) > _F0_MEDIAN_KERNEL:
        f0_voiced = f0[voiced_idx]
        f0_voiced = medfilt(f0_voiced, kernel_size=_F0_MEDIAN_KERNEL)
        f0_voiced = gaussian_filter1d(f0_voiced, sigma=_F0_SIGMA)
        f0_smooth[voiced_idx] = np.clip(f0_voiced, 50.0, 1100.0)

    sp_smooth = gaussian_filter1d(sp, sigma=_SP_SIGMA, axis=0)

    audio_out = pw.synthesize(f0_smooth, sp_smooth, ap, sr, frame_period=_FRAME_PERIOD)
    audio_out = audio_out.astype(np.float32)

    out_rms = np.sqrt(np.mean(audio_out ** 2) + 1e-12)
    audio_out = audio_out * (ref_rms / out_rms)

    min_len = min(len(audio), len(audio_out))
    result = audio.copy()
    result[:min_len] = audio_out[:min_len]

    return result