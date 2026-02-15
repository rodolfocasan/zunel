# zunel/engine.py
import os
import shutil
import librosa
import tempfile
import soundfile
import numpy as np

import torch

from zunel import helpers
from zunel import voice_config
from zunel import audio_analysis
from zunel.architecture import VoiceSynthesizer
from zunel.signal_processing import compute_spectrogram





class SynthBase(object):
    def __init__(self, config_path, device='cuda:0'):
        if 'cuda' in device:
            assert torch.cuda.is_available()

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

    def extract_se(self, ref_wav_list, se_save_path=None, use_perturbation=False):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]

        if use_perturbation:
            try:
                from zunel.perturbation import extract_robust_embedding_multi_sample
                print("[zunel] Using perturbation-based robust embedding extraction...")
                result = extract_robust_embedding_multi_sample(ref_wav_list, self, num_perturbations=3)
                
                if se_save_path is not None:
                    os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
                    torch.save(result.cpu(), se_save_path)
                return result
            except Exception as e:
                print(f"[zunel] Perturbation failed ({e}), falling back to standard extraction")
        
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

        result = torch.stack(embeddings).mean(0)
        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(result.cpu(), se_save_path)
        return result

    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default"):
        cfg = self.cfg
        audio, _ = librosa.load(audio_src_path, sr=cfg.audio.sample_rate)

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device).unsqueeze(0)
            
            spec = compute_spectrogram(
                y, cfg.audio.fft_size, cfg.audio.sample_rate,
                cfg.audio.frame_shift, cfg.audio.frame_length, center=False,
            ).to(self.device)
            
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][0, 0].data.cpu().float().numpy()

        if output_path is None:
            return audio
        soundfile.write(output_path, audio, cfg.audio.sample_rate)

    def compute_embedding_similarity(self, emb1, emb2):
        emb1_norm = emb1 / (torch.norm(emb1, dim=0, keepdim=True) + 1e-8)
        emb2_norm = emb2 / (torch.norm(emb2, dim=0, keepdim=True) + 1e-8)
        similarity = torch.sum(emb1_norm * emb2_norm)
        return similarity.item()





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

    async def generate_source_embedding(self, target_language, gender, voice_version=0):
        voice = voice_config.get_voice(target_language, gender, voice_version)
        calibration_texts = voice_config.get_calibration_texts(target_language)
        
        ref_paths = []
        for i, text in enumerate(calibration_texts):
            ref_path = os.path.join(self.temp_dir, f'ref_{target_language}_{gender}_{i}.wav')
            await self.tts_generator.save_with_fallback(
                text = text,
                preferred_voice = voice,
                output_file = ref_path,
            )
            ref_paths.append(ref_path)
            print(f"[zunel] Generated calibration sample {i + 1}/{len(calibration_texts)}")
        
        embedding_path = os.path.join(self.temp_dir, f'embedding_{target_language}_{gender}.pth')
        embedding = self.converter.extract_se(ref_paths, se_save_path=embedding_path, use_perturbation=False)
        print(f"[zunel] Created embedding for {target_language}/{gender}")
        return embedding, ref_paths[0]

    async def clone_voice(
        self,
        reference_audio_path,
        target_language,
        target_text,
        gender,
        output_path,
        voice_version = 0,
        auto_params = True,
        manual_pitch = None,
        manual_speed = None,
        manual_volume = None,
        tau = None,
        use_perturbation = False
    ):
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"[zunel] Reference audio not found: {reference_audio_path}")
        
        print(f"[zunel] Starting voice cloning process...")
        print(f"[zunel] Reference: {reference_audio_path}")
        print(f"[zunel] Target language: {target_language}")
        print(f"[zunel] Gender: {gender}")
        
        if use_perturbation:
            print("[zunel] Extracting robust target speaker embedding with perturbation...")
        else:
            print("[zunel] Extracting target speaker embedding...")
        target_se = self.converter.extract_se([reference_audio_path], use_perturbation=use_perturbation)
        
        if auto_params:
            print("[zunel] Analyzing reference audio...")
            params = audio_analysis.analyze_audio(reference_audio_path)
            
            pitch_hz = params['pitch_hz']
            detected_gender = params.get('detected_gender')
            is_extreme = params.get('is_extreme_voice', False)
            recommended_tau = params.get('recommended_tau', 0.3)
            speech_rate = params.get('speech_rate_sps', 0)
            loudness = params.get('loudness_lufs', -20)
            
            print(f"[zunel] Detected pitch: {pitch_hz} Hz ({detected_gender})")
            print(f"[zunel] Speech rate: {speech_rate:.2f} syllables/sec")
            print(f"[zunel] Loudness: {loudness:.1f} LUFS")
            
            if detected_gender and detected_gender != gender:
                print(f"[zunel] WARNING: Detected gender '{detected_gender}' differs from specified '{gender}'")
                print(f"[zunel] Consider using gender='{detected_gender}' for better results")
            
            if is_extreme:
                print(f"[zunel] WARNING: Detected extreme voice characteristics")
                print(f"[zunel] Adjusting tau to {recommended_tau} for better quality")
            
            if tau is None:
                tau = recommended_tau
            
            pitch = 0
            speed = 0
            volume = 0
            
            print(f"[zunel] Using neutral TTS parameters (pitch=0Hz, speed=0%, volume=0%)")
            print(f"[zunel] Voice conversion will handle timbre transfer with tau={tau}")
        else:
            pitch = manual_pitch if manual_pitch is not None else 0
            speed = manual_speed if manual_speed is not None else 0
            volume = manual_volume if manual_volume is not None else 0
            
            if tau is None:
                tau = 0.3
            
            print(f"[zunel] Using manual params: pitch={pitch:+d}Hz, speed={speed:+d}%, volume={volume:+d}%")
            print(f"[zunel] Using tau={tau}")
        
        source_se, _ = await self.generate_source_embedding(target_language, gender, voice_version)
        
        similarity = self.converter.compute_embedding_similarity(source_se, target_se)
        print(f"[zunel] Source-Target embedding similarity: {similarity:.4f}")
        
        voice = voice_config.get_voice(target_language, gender, voice_version)
        tmp_synthesis_path = os.path.join(self.temp_dir, 'tmp_synthesis.wav')
        
        await self.tts_generator.save(
            text = target_text,
            voice = voice,
            pitch = f"{pitch:+d}Hz" if pitch != 0 else "+0Hz",
            rate = f"{speed:+d}%" if speed != 0 else "+0%",
            volume = f"{volume:+d}%" if volume != 0 else "+0%",
            output_file = tmp_synthesis_path
        )
        
        print("[zunel] Performing voice conversion...")
        self.converter.convert(
            audio_src_path = tmp_synthesis_path,
            src_se = source_se,
            tgt_se = target_se,
            output_path = output_path,
            tau = tau
        )
        print(f"[zunel] Voice cloning complete: {output_path}")
        return output_path