# zunel/processing.py
import librosa
import numpy as np
import soundfile as sf
from scipy import signal





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