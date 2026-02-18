# zunel/engine.py
import os
import random
import shutil
import librosa
import tempfile
import soundfile
import numpy as np

import torch

from zunel import helpers
from zunel import voice_config
from zunel.architecture import VoiceSynthesizer
from zunel.signal_processing import compute_spectrogram
from zunel.adapters import SpeakerAdapter
from zunel.processing import enhance_tts, remove_voice_conversion_artifacts





class SynthBase(object):
    def __init__(self, config_path, device='auto'):
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif 'cuda' in device and not torch.cuda.is_available():
            print(f"[zunel] WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        cfg = helpers.load_config(config_path)

        model = VoiceSynthesizer(
            len(getattr(cfg, 'symbols', [])),
            cfg.audio.fft_size // 2 + 1,
            n_speakers = cfg.audio.num_speakers,
            **cfg.architecture,
        ).to(device)
        model.eval()

        self.model = model
        self.cfg = cfg
        self.device = device

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.device))
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)
        print("[zunel] Loaded checkpoint '{}'".format(ckpt_path))
        print('[zunel] missing/unexpected keys:', missing, unexpected)





class TimbreConverter(SynthBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = getattr(self.cfg, '_release_', "1.0.0")
        self.speaker_adapter_src = None
        self.speaker_adapter_tgt = None

    def load_adapters(self, adapter_path):
        if not os.path.exists(adapter_path):
            print(f"[zunel] No adapters found at {adapter_path}")
            return

        checkpoint = torch.load(adapter_path, map_location=self.device, weights_only=False)
        embedding_dim = getattr(self.cfg.architecture, 'embedding_dim', 256)

        self.speaker_adapter_src = SpeakerAdapter(embedding_dim).to(self.device)
        self.speaker_adapter_tgt = SpeakerAdapter(embedding_dim).to(self.device)

        self.speaker_adapter_src.load_state_dict(checkpoint['adapter_src'])
        self.speaker_adapter_tgt.load_state_dict(checkpoint['adapter_tgt'])

        self.speaker_adapter_src.eval()
        self.speaker_adapter_tgt.eval()

        self.model.set_speaker_adapters(self.speaker_adapter_src, self.speaker_adapter_tgt)
        print(f"[zunel] Loaded speaker adapters from {adapter_path}")

    def extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]

        embeddings = []
        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=self.cfg.audio.sample_rate)
            y = torch.FloatTensor(audio_ref).to(self.device).unsqueeze(0)

            spec = compute_spectrogram(
                y, self.cfg.audio.fft_size, self.cfg.audio.sample_rate,
                self.cfg.audio.frame_shift, self.cfg.audio.frame_length, center=False,
            ).to(self.device)

            with torch.no_grad():
                g = self.model.speaker_embedder(spec.transpose(1, 2)).unsqueeze(-1)
                embeddings.append(g.detach())

        embedding = torch.stack(embeddings).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(embedding.cpu(), se_save_path)

        return embedding

    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3):
        cfg = self.cfg
        audio, _ = librosa.load(audio_src_path, sr=cfg.audio.sample_rate)

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device).unsqueeze(0)

            spec = compute_spectrogram(
                y, cfg.audio.fft_size, cfg.audio.sample_rate,
                cfg.audio.frame_shift, cfg.audio.frame_length, center=False,
            ).to(self.device)

            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio_out = self.model.voice_conversion(
                spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau
            )[0][0, 0].data.cpu().float().numpy()

        audio_out = remove_voice_conversion_artifacts(
            audio_out,
            sr=cfg.audio.sample_rate,
            n_fft=cfg.audio.fft_size,
            hop_length=cfg.audio.frame_shift
        )

        if output_path is None:
            return audio_out
        soundfile.write(output_path, audio_out, cfg.audio.sample_rate)





class VoiceCloner:
    def __init__(self, converter, tts_generator):
        self.converter = converter
        self.tts_generator = tts_generator
        self.temp_dir = tempfile.mkdtemp(prefix='zunel_')
        print(f"[zunel] Temporary directory created: {self.temp_dir}")

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[zunel] Temporary directory cleaned: {self.temp_dir}")

    async def clone_voice(
        self,
        reference_audio_path,
        target_language,
        target_text,
        gender,
        output_path,
        voice_version = 0,
        tau = 0.3
    ):
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"[zunel] Reference audio not found: {reference_audio_path}")

        print(f"[zunel] Starting voice cloning...")
        print(f"[zunel] Reference: {reference_audio_path}")
        print(f"[zunel] Target language: {target_language} | Gender: {gender} | Tau: {tau}")

        voice = voice_config.get_voice(target_language, gender, voice_version)

        tts_raw_path = os.path.join(self.temp_dir, 'tts_raw.wav')
        tts_enhanced_path = os.path.join(self.temp_dir, 'tts_enhanced.wav')

        await self.tts_generator.save(
            text = target_text,
            voice = voice,
            pitch = "+0Hz",
            rate = "+0%",
            volume = "+0%",
            output_file = tts_raw_path
        )

        enhance_tts(tts_raw_path, tts_enhanced_path, sr=self.converter.cfg.audio.sample_rate)

        print("[zunel] Extracting embeddings...")
        target_se = self.converter.extract_se([reference_audio_path])
        source_se = self.converter.extract_se([tts_enhanced_path])

        print("[zunel] Performing voice conversion...")
        self.converter.convert(
            audio_src_path = tts_enhanced_path,
            src_se = source_se,
            tgt_se = target_se,
            output_path = output_path,
            tau = tau
        )
        print(f"[zunel] Voice cloning complete: {output_path}")
        return output_path