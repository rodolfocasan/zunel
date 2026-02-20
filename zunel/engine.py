# zunel/engine.py
import os
import platform
import random
import shutil
import librosa
import tempfile
import soundfile
import numpy as np

import torch

from zunel.core import helpers
from zunel.voices import voice_config
from zunel.core.architecture import VoiceSynthesizer
from zunel.audio.signal_processing import compute_spectrogram
from zunel.core.adapters import SpeakerAdapter
from zunel.audio.processing import enhance_tts
from zunel.utils.resources import resolve_thread_counts





_MAX_REF_DURATION = 8.0


def _remove_all_weight_norm(model):
    from torch.nn.utils import remove_weight_norm

    try:
        model.wave_decoder.remove_weight_norm()
    except Exception:
        pass

    for conv in model.speaker_embedder.convs:
        try:
            remove_weight_norm(conv)
        except Exception:
            pass

    try:
        model.var_encoder.enc.remove_weight_norm()
    except Exception:
        pass

    for flow in model.norm_flow.flows:
        if hasattr(flow, 'enc') and hasattr(flow.enc, 'remove_weight_norm'):
            try:
                flow.enc.remove_weight_norm()
            except Exception:
                pass
    print('[zunel] Weight norm removed from WaveDecoder, SpeakerEmbedder, VariationalEncoder, NormalizingFlow')


def _set_quantized_backend():
    machine = platform.machine().lower()

    if machine in ('aarch64', 'arm64', 'armv7l'):
        torch.backends.quantized.engine = 'qnnpack'
    else:
        torch.backends.quantized.engine = 'x86'
    print(f'[zunel] Quantized backend: {torch.backends.quantized.engine} (arch: {machine})')


def _apply_threads(mode):
    total = os.cpu_count() or 1
    n_intra, n_interop = resolve_thread_counts(mode)

    torch.set_num_threads(n_intra)

    try:
        torch.set_num_interop_threads(n_interop)
    except RuntimeError:
        pass
    print(f'[zunel] Thread mode: {mode} | intra: {n_intra} | interop: {n_interop} | total CPUs: {total}')


def _quantize_linear_torchao(model):
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig())
        return True
    except ImportError:
        pass

    try:
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(model, int8_weight_only())
        return True
    except (ImportError, Exception):
        pass
    return False


def _quantize_gru_legacy(model):
    torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.GRU},
        dtype = torch.qint8,
        inplace = True
    )


def _quantize_all_legacy(model):
    torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.GRU},
        dtype = torch.qint8,
        inplace = True
    )





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
        self._se_cache = {}

    def optimize_for_cpu(self, quantize=True, compile_model=True, thread_mode='deterministic'):
        """
        thread_mode options:
            'deterministic' -> 1 thread, bit-identical results across any machine/run
            'max_speed'     -> all available threads, fastest inference, consistent per machine
        """
        if 'cuda' in str(self.device):
            self.model = self.model.cpu()
            self.device = 'cpu'

        self.model.eval()
        _remove_all_weight_norm(self.model)
        _apply_threads(thread_mode)
        torch.set_float32_matmul_precision('high')

        if quantize:
            _set_quantized_backend()

            ao_ok = _quantize_linear_torchao(self.model)
            if ao_ok:
                _quantize_gru_legacy(self.model)
            else:
                _quantize_all_legacy(self.model)

        if compile_model and hasattr(torch, 'compile'):
            try:
                os.environ.setdefault('TORCHINDUCTOR_FREEZING', '1')
                self.model = torch.compile(
                    self.model,
                    mode = 'reduce-overhead',
                    dynamic = True,
                )
                print('[zunel] Model compiled with torch.compile (reduce-overhead, dynamic=True)')
            except Exception as e:
                print(f'[zunel] torch.compile skipped: {e}')

    def warmup(self):
        cfg = self.cfg
        spec_channels = cfg.audio.fft_size // 2 + 1
        dummy_spec = torch.zeros(1, spec_channels, 100, device=self.device)
        dummy_lengths = torch.LongTensor([100]).to(self.device)
        dummy_se = torch.zeros(1, cfg.architecture.embedding_dim, 1, device=self.device)

        print('[zunel] Running warmup pass...')
        with torch.inference_mode():
            try:
                self.model.voice_conversion(
                    dummy_spec,
                    dummy_lengths,
                    sid_src = dummy_se,
                    sid_tgt = dummy_se,
                    tau = 0.3,
                )
                self.model.speaker_embedder(dummy_spec.transpose(1, 2))
            except Exception as e:
                print(f'[zunel] Warmup partial: {e}')
        print('[zunel] Warmup complete')

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

    def _spec_from_path(self, path, max_duration=None):
        cfg = self.cfg
        audio, _ = librosa.load(
            path,
            sr = cfg.audio.sample_rate,
            duration = max_duration,
            mono = True,
        )
        y = torch.FloatTensor(audio).to(self.device).unsqueeze(0)
        spec = compute_spectrogram(
            y, cfg.audio.fft_size, cfg.audio.sample_rate,
            cfg.audio.frame_shift, cfg.audio.frame_length, center=False,
        ).to(self.device)
        return spec

    def _se_from_spec(self, spec):
        with torch.inference_mode():
            return self.model.speaker_embedder(spec.transpose(1, 2)).unsqueeze(-1)

    def extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]

        embeddings = []
        for fname in ref_wav_list:
            try:
                cache_key = (fname, os.path.getmtime(fname))
            except OSError:
                cache_key = None

            if cache_key is not None and cache_key in self._se_cache:
                embeddings.append(self._se_cache[cache_key])
                continue

            spec = self._spec_from_path(fname, max_duration=_MAX_REF_DURATION)
            emb = self._se_from_spec(spec).cpu()

            if cache_key is not None:
                self._se_cache[cache_key] = emb
            embeddings.append(emb)

        embedding = torch.stack(embeddings).mean(0).to(self.device)

        if se_save_path is not None:
            parent = os.path.dirname(se_save_path)
            
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(embedding.cpu(), se_save_path)
        return embedding

    def _convert_from_spec(self, spec, src_se, tgt_se, tau=0.3):
        cfg = self.cfg
        with torch.inference_mode():
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = self.model.voice_conversion(
                spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau
            )[0][0, 0].data.cpu().float().numpy()
        return audio

    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3):
        spec = self._spec_from_path(audio_src_path)
        audio = self._convert_from_spec(spec, src_se, tgt_se, tau=tau)

        if output_path is None:
            return audio

        soundfile.write(output_path, audio, self.cfg.audio.sample_rate)





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

        tts_spec = self.converter._spec_from_path(tts_enhanced_path)
        source_se = self.converter._se_from_spec(tts_spec)

        print("[zunel] Performing voice conversion...")
        audio = self.converter._convert_from_spec(tts_spec, source_se, target_se, tau=tau)
        soundfile.write(output_path, audio, self.converter.cfg.audio.sample_rate)

        print(f"[zunel] Voice cloning complete: {output_path}")
        return output_path