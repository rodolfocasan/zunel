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
from zunel.bottleneck import BottleneckModule
from zunel.adapters import SpeakerAdapter





class SynthBase(object):
    def __init__(self, config_path, device='auto'):
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif 'cuda' in device and not torch.cuda.is_available():
            print(f"[zunel] WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'

        cfg = helpers.load_config(config_path)

        model = VoiceSynthesizer(
            len(getattr(cfg, 'symbols', [])),
            cfg.audio.fft_size // 2 + 1,
            n_speakers=cfg.audio.num_speakers,
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
    def __init__(self, *args, bottleneck_type='segment_gst', **kwargs):
        super().__init__(*args, **kwargs)
        self.version = getattr(self.cfg, '_release_', "1.0.0")
        
        embedding_dim = getattr(self.cfg.architecture, 'embedding_dim', 256)
        
        self.bottleneck = BottleneckModule(
            input_dim=embedding_dim,
            bottleneck_dim=1024,
            output_dim=embedding_dim,
            bottleneck_type=bottleneck_type,
            n_style_tokens=10,
            n_heads=8,
            n_vertices=128,
            kl_weight=0.0001
        ).to(self.device)
        
        self.bottleneck.eval()
        
        self.speaker_adapter_src = None
        self.speaker_adapter_tgt = None
        
        print(f"[zunel] Initialized {bottleneck_type} bottleneck for voice transfer")

    def load_adapters(self, adapter_path):
        if os.path.exists(adapter_path):
            checkpoint = torch.load(adapter_path, map_location=self.device)
            
            embedding_dim = getattr(self.cfg.architecture, 'embedding_dim', 256)
            
            self.speaker_adapter_src = SpeakerAdapter(embedding_dim).to(self.device)
            self.speaker_adapter_tgt = SpeakerAdapter(embedding_dim).to(self.device)
            
            self.speaker_adapter_src.load_state_dict(checkpoint['adapter_src'])
            self.speaker_adapter_tgt.load_state_dict(checkpoint['adapter_tgt'])
            
            self.speaker_adapter_src.eval()
            self.speaker_adapter_tgt.eval()
            
            self.model.set_speaker_adapters(self.speaker_adapter_src, self.speaker_adapter_tgt)
            
            print(f"[zunel] Loaded speaker adapters from {adapter_path}")
        else:
            print(f"[zunel] No adapters found at {adapter_path}, using base model")

    def extract_se(self, ref_wav_list, se_save_path=None, use_bottleneck=True):
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

        raw_embedding = torch.stack(embeddings).mean(0)
        
        if use_bottleneck:
            with torch.no_grad():
                refined_embedding, kl_loss = self.bottleneck(raw_embedding, inference=True)
                result = refined_embedding
                print(f"[zunel] Applied bottleneck refinement to speaker embedding")
        else:
            result = raw_embedding
            print(f"[zunel] Using raw speaker embedding (no bottleneck)")
        
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


def compute_adaptive_tau(reference_params, source_embedding, target_embedding):
    formant_dispersion = reference_params.get('formant_dispersion', 0)
    spectral_complexity = reference_params.get('spectral_envelope_complexity', 0)
    pitch_variability = reference_params.get('pitch_variability', 0)
    
    jitter = reference_params.get('jitter_local', 0)
    shimmer = reference_params.get('shimmer_local', 0)
    hnr = reference_params.get('hnr', 20)
    roughness_score = reference_params.get('roughness_score', 0)
    
    low_band_energy = reference_params.get('low_band_energy', 0)
    mid_band_energy = reference_params.get('mid_band_energy', 0)
    high_band_energy = reference_params.get('high_band_energy', 0)
    spectral_flux = reference_params.get('spectral_flux', 0)
    
    embedding_distance = compute_embedding_distance(source_embedding, target_embedding)
    
    voice_complexity_score = 0.0
    
    if formant_dispersion > 0:
        voice_complexity_score += min(formant_dispersion / 1500.0, 1.0) * 0.20
    if spectral_complexity > 0:
        voice_complexity_score += min(spectral_complexity / 100.0, 1.0) * 0.20
    if pitch_variability > 0:
        voice_complexity_score += min(pitch_variability / 50.0, 1.0) * 0.15
    
    if roughness_score > 0.3:
        voice_complexity_score += roughness_score * 0.25
    
    if jitter > 0.005:
        voice_complexity_score += min((jitter - 0.005) / 0.01, 1.0) * 0.10
    
    if shimmer > 0.03:
        voice_complexity_score += min((shimmer - 0.03) / 0.05, 1.0) * 0.10
    
    if hnr > 0 and hnr < 15:
        hnr_penalty = (15.0 - hnr) / 15.0
        voice_complexity_score += hnr_penalty * 0.15
    
    spectral_balance = 0.0
    total_energy = low_band_energy + mid_band_energy + high_band_energy
    if total_energy > 0:
        spectral_balance = abs(high_band_energy / total_energy - 0.33)
        voice_complexity_score += spectral_balance * 0.10
    
    if spectral_flux > 0:
        flux_normalized = min(spectral_flux / 1e6, 1.0)
        voice_complexity_score += flux_normalized * 0.08
    
    embedding_similarity_bonus = max(0, 1.0 - embedding_distance * 2.0) * 0.12
    voice_complexity_score += embedding_similarity_bonus
    
    base_tau = 0.30
    complexity_adjustment = voice_complexity_score * 0.40
    distance_adjustment = embedding_distance * 0.25
    
    adaptive_tau = base_tau + complexity_adjustment + distance_adjustment
    
    confidence_factors = {
        'formant_strength': 1.0 if formant_dispersion > 500 else 0.5,
        'spectral_quality': 1.0 if spectral_complexity > 10 else 0.5,
        'prosodic_stability': 1.0 if pitch_variability > 10 else 0.5,
        'voice_quality': 1.0 if (jitter > 0 and shimmer > 0 and hnr > 0) else 0.3
    }
    
    confidence_score = sum(confidence_factors.values()) / len(confidence_factors)
    
    if confidence_score < 0.6:
        adaptive_tau = min(adaptive_tau + 0.10, 0.95)
    
    if roughness_score > 0.5:
        roughness_boost = (roughness_score - 0.5) * 0.80
        adaptive_tau = min(adaptive_tau + roughness_boost, 0.95)
        print(f"[zunel] Detected rough/raspy voice, applying roughness boost: +{roughness_boost:.3f}")
    
    adaptive_tau = max(0.20, min(adaptive_tau, 0.95))
    
    return adaptive_tau, embedding_distance, voice_complexity_score, roughness_score





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

    async def generate_source_embedding(self, target_language, gender, voice_version=0, use_bottleneck=True):
        voice = voice_config.get_voice(target_language, gender, voice_version)
        calibration_texts = voice_config.get_calibration_texts(target_language)
        
        ref_paths = []
        for i, text in enumerate(calibration_texts):
            ref_path = os.path.join(self.temp_dir, f'ref_{target_language}_{gender}_{i}.wav')
            await self.tts_generator.save_with_fallback(
                text=text,
                preferred_voice=voice,
                output_file=ref_path,
            )
            ref_paths.append(ref_path)
            print(f"[zunel] Generated calibration sample {i + 1}/{len(calibration_texts)}")
        
        embedding_path = os.path.join(self.temp_dir, f'embedding_{target_language}_{gender}.pth')
        embedding = self.converter.extract_se(ref_paths, se_save_path=embedding_path, use_bottleneck=use_bottleneck)
        print(f"[zunel] Created refined embedding for {target_language}/{gender}")
        return embedding, ref_paths[0]

    async def clone_voice(
        self,
        reference_audio_path,
        target_language,
        target_text,
        gender,
        output_path,
        voice_version=0,
        auto_params=True,
        manual_pitch=None,
        manual_speed=None,
        manual_volume=None,
        use_bottleneck=False
    ):
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"[zunel] Reference audio not found: {reference_audio_path}")
        
        print(f"[zunel] Starting voice cloning...")
        print(f"[zunel] Reference: {reference_audio_path}")
        print(f"[zunel] Target language: {target_language}")
        print(f"[zunel] Gender: {gender}")
        print(f"[zunel] Bottleneck: {'ENABLED' if use_bottleneck else 'DISABLED (recommended for better identity preservation)'}")
        
        print("[zunel] Extracting target speaker embedding...")
        target_se = self.converter.extract_se([reference_audio_path], use_bottleneck=use_bottleneck)
        
        params = None
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
            spectral_centroid = params.get('spectral_centroid', 0)
            
            if spectral_complexity > 0:
                print(f"[zunel] Spectral envelope complexity: {spectral_complexity:.2f}")
            if spectral_centroid > 0:
                print(f"[zunel] Spectral centroid: {spectral_centroid:.0f} Hz")
            
            jitter = params.get('jitter_local', 0)
            shimmer = params.get('shimmer_local', 0)
            hnr = params.get('hnr', 0)
            roughness_score = params.get('roughness_score', 0)
            
            if jitter > 0 or shimmer > 0 or hnr > 0:
                print(f"[zunel] Voice quality: Jitter={jitter:.4f}, Shimmer={shimmer:.4f}, HNR={hnr:.2f}dB")
                print(f"[zunel] Roughness score: {roughness_score:.3f}")
            
            if roughness_score > 0.4:
                print(f"[zunel] Detected rough/raspy voice characteristics - will use high tau")
            
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
        
        print("[zunel] Generating source embedding...")
        source_se, _ = await self.generate_source_embedding(target_language, gender, voice_version, use_bottleneck=use_bottleneck)
        
        if auto_params and params:
            adaptive_tau, emb_distance, complexity_score, rough_score = compute_adaptive_tau(
                params, source_se, target_se
            )
            
            print(f"[zunel] Embedding distance: {emb_distance:.4f}")
            print(f"[zunel] Voice complexity score: {complexity_score:.4f}")
            print(f"[zunel] Roughness score: {rough_score:.4f}")
            print(f"[zunel] Adaptive tau: {adaptive_tau:.4f}")
            tau = adaptive_tau
        else:
            tau = 0.50
            print(f"[zunel] Using default tau: {tau:.4f}")
        
        voice = voice_config.get_voice(target_language, gender, voice_version)
        tmp_synthesis_path = os.path.join(self.temp_dir, 'tmp_synthesis.wav')
        
        await self.tts_generator.save(
            text=target_text,
            voice=voice,
            pitch=f"{pitch:+d}Hz" if pitch != 0 else "+0Hz",
            rate=f"{speed:+d}%" if speed != 0 else "+0%",
            volume=f"{volume:+d}%" if volume != 0 else "+0%",
            output_file=tmp_synthesis_path
        )
        
        print("[zunel] Performing voice conversion...")
        self.converter.convert(
            audio_src_path=tmp_synthesis_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            tau=tau
        )
        print(f"[zunel] Voice cloning complete: {output_path}")
        return output_path