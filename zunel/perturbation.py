# zunel/perturbation.py
import librosa
import numpy as np
import soundfile as sf
from scipy import signal

import torch

from zunel.signal_processing import compute_spectrogram





class AudioPerturbation:
    @staticmethod
    def pitch_shift(audio, sr, n_steps):
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    @staticmethod
    def time_stretch(audio, rate):
        return librosa.effects.time_stretch(audio, rate=rate)

    @staticmethod
    def formant_shift(audio, sr, shift_factor):
        n_fft = 2048
        hop_length = n_fft // 4
        
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(D), np.angle(D)
        
        freq_bins = mag.shape[0]
        shift_bins = int(freq_bins * (shift_factor - 1.0))
        
        if shift_bins > 0:
            mag_shifted = np.zeros_like(mag)
            mag_shifted[shift_bins:] = mag[:-shift_bins]
        elif shift_bins < 0:
            mag_shifted = np.zeros_like(mag)
            mag_shifted[:shift_bins] = mag[-shift_bins:]
        else:
            mag_shifted = mag
        
        D_shifted = mag_shifted * np.exp(1j * phase)
        audio_shifted = librosa.istft(D_shifted, hop_length=hop_length)
        return audio_shifted

    @staticmethod
    def apply_augmentation(audio, sr, config=None):
        if config is None:
            config = {
                'pitch_shift': np.random.uniform(-2, 2),
                'formant_shift': np.random.uniform(0.95, 1.05)
            }
        
        if 'pitch_shift' in config and abs(config['pitch_shift']) > 0.1:
            audio = AudioPerturbation.pitch_shift(audio, sr, config['pitch_shift'])
        
        if 'formant_shift' in config and abs(config['formant_shift'] - 1.0) > 0.01:
            audio = AudioPerturbation.formant_shift(audio, sr, config['formant_shift'])
        return audio


def extract_robust_embedding_multi_sample(audio_paths, converter, num_perturbations=3):
    all_embeddings = []
    
    for audio_path in audio_paths:
        audio, sr = librosa.load(audio_path, sr=converter.cfg.audio.sample_rate)
        
        embeddings_for_sample = []
        
        y_orig = torch.FloatTensor(audio).to(converter.device).unsqueeze(0)
        spec_orig = compute_spectrogram(
            y_orig, converter.cfg.audio.fft_size, converter.cfg.audio.sample_rate,
            converter.cfg.audio.frame_shift, converter.cfg.audio.frame_length, center=False
        ).to(converter.device)
        
        with torch.no_grad():
            emb_orig = converter.model.speaker_embedder(spec_orig.transpose(1, 2))
            embeddings_for_sample.append(emb_orig)
        
        for _ in range(num_perturbations):
            perturbed_audio = AudioPerturbation.apply_augmentation(audio, sr)
            
            if len(perturbed_audio) < len(audio):
                perturbed_audio = np.pad(perturbed_audio, (0, len(audio) - len(perturbed_audio)))
            elif len(perturbed_audio) > len(audio):
                perturbed_audio = perturbed_audio[:len(audio)]
            
            y_pert = torch.FloatTensor(perturbed_audio).to(converter.device).unsqueeze(0)
            spec_pert = compute_spectrogram(
                y_pert, converter.cfg.audio.fft_size, converter.cfg.audio.sample_rate,
                converter.cfg.audio.frame_shift, converter.cfg.audio.frame_length, center=False
            ).to(converter.device)
            
            with torch.no_grad():
                emb_pert = converter.model.speaker_embedder(spec_pert.transpose(1, 2))
                embeddings_for_sample.append(emb_pert)
        
        sample_emb = torch.stack([e.squeeze(0) for e in embeddings_for_sample]).mean(0)
        all_embeddings.append(sample_emb)
    
    final_embedding = torch.stack(all_embeddings).mean(0).unsqueeze(-1)
    return final_embedding