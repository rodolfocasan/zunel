# zunel/engine.py
import os
import shutil
import librosa
import tempfile
import soundfile
import numpy as np

import torch
import torch.nn.functional as F

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


def compute_embedding_distance(embedding1, embedding2):
    if embedding1.dim() == 3:
        embedding1 = embedding1.mean(dim=-1)
    if embedding2.dim() == 3:
        embedding2 = embedding2.mean(dim=-1)
    
    cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=1)
    distance = 1.0 - cosine_sim.mean().item()
    return distance


def compute_comprehensive_tau(reference_params, source_embedding, target_embedding):
    formant_dispersion = reference_params.get('formant_dispersion', 0)
    spectral_complexity = reference_params.get('spectral_envelope_complexity', 0)
    pitch_variability = reference_params.get('pitch_variability', 0)
    
    roughness_score = reference_params.get('roughness_score', 0)
    is_rough_voice = reference_params.get('is_rough_voice', False)
    jitter_percent = reference_params.get('jitter_percent', 0)
    shimmer_percent = reference_params.get('shimmer_percent', 0)
    hnr = reference_params.get('hnr', 15.0)
    
    estimated_age = reference_params.get('estimated_age', 30.0)
    age_score = reference_params.get('age_score', 0.0)
    vtln_warp_factor = reference_params.get('vtln_warp_factor', 1.0)
    
    embedding_distance = compute_embedding_distance(source_embedding, target_embedding)
    
    voice_complexity_score = 0.0
    if formant_dispersion > 0:
        voice_complexity_score += min(formant_dispersion / 1500.0, 1.0) * 0.25
    if spectral_complexity > 0:
        voice_complexity_score += min(spectral_complexity / 100.0, 1.0) * 0.25
    if pitch_variability > 0:
        voice_complexity_score += min(pitch_variability / 50.0, 1.0) * 0.15
    
    embedding_similarity_bonus = max(0, 1.0 - embedding_distance * 2.0) * 0.15
    voice_complexity_score += embedding_similarity_bonus
    
    roughness_adjustment = 0.0
    if is_rough_voice:
        roughness_intensity = roughness_score
        
        if jitter_percent > 2.0:
            roughness_intensity += min((jitter_percent - 2.0) / 3.0, 0.5)
        if shimmer_percent > 8.0:
            roughness_intensity += min((shimmer_percent - 8.0) / 7.0, 0.5)
        if hnr < 10.0:
            roughness_intensity += min((10.0 - hnr) / 6.0, 0.5)
        
        roughness_intensity = min(roughness_intensity, 1.5)
        
        roughness_adjustment = roughness_intensity * 0.12
    
    age_adjustment = 0.0
    if estimated_age > 35.0:
        age_factor = (estimated_age - 35.0) / 45.0
        age_adjustment = min(age_factor, 1.0) * 0.10
    
    vtln_adjustment = 0.0
    if abs(vtln_warp_factor - 1.0) > 0.05:
        vtln_deviation = abs(vtln_warp_factor - 1.0)
        vtln_adjustment = min(vtln_deviation / 0.20, 1.0) * 0.08
    
    base_tau = 0.08
    complexity_component = voice_complexity_score * 0.18
    distance_component = embedding_distance * 0.22
    roughness_component = roughness_adjustment
    age_component = age_adjustment
    vtln_component = vtln_adjustment
    
    adaptive_tau = base_tau + complexity_component + distance_component + roughness_component + age_component + vtln_component
    
    adaptive_tau = max(0.05, min(adaptive_tau, 0.60))
    
    confidence_factors = {
        'formant_strength': 1.0 if formant_dispersion > 500 else 0.6,
        'spectral_quality': 1.0 if spectral_complexity > 10 else 0.6,
        'prosodic_stability': 1.0 if pitch_variability > 10 else 0.6,
        'voice_quality_valid': 1.0 if hnr > 5.0 else 0.5
    }
    
    confidence_score = sum(confidence_factors.values()) / len(confidence_factors)
    
    if confidence_score < 0.65:
        adaptive_tau = min(adaptive_tau + 0.06, 0.60)
    return adaptive_tau, embedding_distance, voice_complexity_score, roughness_adjustment, age_adjustment





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
        embedding = self.converter.extract_se(ref_paths, se_save_path=embedding_path)
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
        manual_volume = None
    ):
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"[zunel] Reference audio not found: {reference_audio_path}")
        
        print(f"[zunel] Starting voice cloning process...")
        print(f"[zunel] Reference: {reference_audio_path}")
        print(f"[zunel] Target language: {target_language}")
        print(f"[zunel] Gender: {gender}")
        
        target_se = self.converter.extract_se([reference_audio_path])
        
        if auto_params:
            print("[zunel] Analyzing reference audio...")
            params = audio_analysis.analyze_audio(reference_audio_path)
            
            pitch_hz = params['pitch_hz']
            detected_gender = params.get('detected_gender')
            is_extreme = params.get('is_extreme_voice', False)
            speech_rate = params.get('speech_rate_sps', 0)
            loudness = params.get('loudness_lufs', -20)
            
            print(f"[zunel] Detected pitch: {pitch_hz} Hz ({detected_gender})")
            print(f"[zunel] Speech rate: {speech_rate:.2f} syllables/sec")
            print(f"[zunel] Loudness: {loudness:.1f} LUFS")
            
            f1 = params.get('formant_f1_mean', 0)
            f2 = params.get('formant_f2_mean', 0)
            f3 = params.get('formant_f3_mean', 0)
            formant_dispersion = params.get('formant_dispersion', 0)
            
            if f1 > 0 and f2 > 0:
                print(f"[zunel] Formants: F1={f1:.0f}Hz, F2={f2:.0f}Hz, F3={f3:.0f}Hz")
                print(f"[zunel] Formant dispersion: {formant_dispersion:.2f}")
            
            spectral_complexity = params.get('spectral_envelope_complexity', 0)
            if spectral_complexity > 0:
                print(f"[zunel] Spectral envelope complexity: {spectral_complexity:.2f}")
            
            jitter_percent = params.get('jitter_percent', 0)
            shimmer_percent = params.get('shimmer_percent', 0)
            hnr = params.get('hnr', 15.0)
            is_rough = params.get('is_rough_voice', False)
            roughness_score = params.get('roughness_score', 0)
            
            if jitter_percent > 0:
                print(f"[zunel] Voice quality: Jitter={jitter_percent:.2f}%, Shimmer={shimmer_percent:.2f}%, HNR={hnr:.1f}dB")
            
            if is_rough:
                print(f"[zunel] Detected rough/hoarse voice (roughness score: {roughness_score:.3f})")
                print(f"[zunel] Applying specialized compensation for voice quality preservation")
            
            estimated_age = params.get('estimated_age', 30.0)
            age_score = params.get('age_score', 0.0)
            vtl = params.get('vocal_tract_length', 0)
            vtln_warp = params.get('vtln_warp_factor', 1.0)
            
            if estimated_age > 0:
                print(f"[zunel] Estimated vocal age: {estimated_age:.0f} years (age score: {age_score:.3f})")
                print(f"[zunel] Vocal tract length: {vtl:.2f}cm, VTLN factor: {vtln_warp:.3f}")
            
            if estimated_age > 50:
                print(f"[zunel] Detected mature voice - applying age-invariant compensation")
            
            if detected_gender and detected_gender != gender:
                print(f"[zunel] WARNING: Detected gender '{detected_gender}' differs from specified '{gender}'")
                print(f"[zunel] Consider using gender='{detected_gender}' for better results")
            
            if is_extreme:
                print(f"[zunel] WARNING: Detected extreme voice characteristics")
            
            pitch = 0
            speed = 0
            volume = 0
            print(f"[zunel] Using neutral TTS parameters (pitch=0Hz, speed=0%, volume=0%)")
        else:
            pitch = manual_pitch if manual_pitch is not None else 0
            speed = manual_speed if manual_speed is not None else 0
            volume = manual_volume if manual_volume is not None else 0
            print(f"[zunel] Using manual params: pitch={pitch:+d}Hz, speed={speed:+d}%, volume={volume:+d}%")
        
        source_se, _ = await self.generate_source_embedding(target_language, gender, voice_version)
        
        if auto_params:
            adaptive_tau, emb_distance, complexity_score, roughness_adj, age_adj = compute_comprehensive_tau(
                params, source_se, target_se
            )
            
            print(f"[zunel] Embedding distance: {emb_distance:.4f}")
            print(f"[zunel] Voice complexity score: {complexity_score:.4f}")
            
            if roughness_adj > 0:
                print(f"[zunel] Roughness adjustment: +{roughness_adj:.4f}")
            if age_adj > 0:
                print(f"[zunel] Age adjustment: +{age_adj:.4f}")
            
            print(f"[zunel] Adaptive tau: {adaptive_tau:.4f}")
            
            tau = adaptive_tau
        else:
            tau = 0.30
            print(f"[zunel] Using default tau: {tau:.4f}")
        
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
        
        print("[zunel] Performing voice conversion with optimized identity preservation...")
        self.converter.convert(
            audio_src_path = tmp_synthesis_path,
            src_se = source_se,
            tgt_se = target_se,
            output_path = output_path,
            tau = tau
        )
        print(f"[zunel] Voice cloning complete: {output_path}")
        return output_path