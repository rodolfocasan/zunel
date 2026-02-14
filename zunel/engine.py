# zunel/engine.py
import os
import re
import librosa
import soundfile
import numpy as np

import torch

from zunel import helpers
from zunel.architecture import VoiceSynthesizer
from zunel.text_processing import encode_text, symbols
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





class NeuralSpeaker(SynthBase):
    language_marks = {
        "english": "EN",
        "chinese": "ZH",
    }

    @staticmethod
    def _build_phone_sequence(text, cfg, is_symbol):
        if is_symbol:
            cleaners = []
        else:
            cleaners = cfg.audio.text_cleaners

        seq = encode_text(text, cfg.symbols, cleaners)
        if cfg.audio.add_blank:
            seq = helpers.interleave_with(seq, 0)
        result = torch.LongTensor(seq)
        return result

    @staticmethod
    def _concat_audio_segments(segments, sr, speed=1.):
        buf = []
        for seg in segments:
            buf += seg.reshape(-1).tolist()
            buf += [0] * int((sr * 0.05) / speed)
        return np.array(buf).astype(np.float32)

    @staticmethod
    def _chunk_text(text, lang_tag):
        chunks = helpers.segment_text(text, language_str=lang_tag)
        print(" [zunel] Text splitted to sentences:")
        print('\n'.join(chunks))
        return chunks

    def tts(self, text, output_path, speaker, language='English', speed=1.0):
        language_key = language.lower()

        if language_key in self.language_marks:
            mark = self.language_marks[language_key]
        else:
            mark = None

        assert mark is not None, "language " + str(language) + " is not supported"

        chunks = self._chunk_text(text, mark)
        audio_list = []
        for chunk in chunks:
            chunk = re.sub(r'([a-z])([A-Z])', r'\1 \2', chunk)
            chunk = "[" + str(mark) + "]" + chunk + "[" + str(mark) + "]"
            phones = self._build_phone_sequence(chunk, self.cfg, False)
            speaker_id = self.cfg.speakers[speaker]

            with torch.no_grad():
                x = phones.unsqueeze(0).to(self.device)

                x_len = torch.LongTensor([phones.size(0)])
                x_len = x_len.to(self.device)

                sid = torch.LongTensor([speaker_id])
                sid = sid.to(self.device)

                audio = self.model.infer(
                    x,
                    x_len,
                    sid = sid,
                    noise_scale = 0.667,
                    noise_scale_w = 0.6,
                    length_scale = 1.0 / speed,
                )

                audio = audio[0][0, 0].data.cpu().float().numpy()
            audio_list.append(audio)
        
        audio = self._concat_audio_segments(
            audio_list,
            sr = self.cfg.audio.sample_rate,
            speed = speed
        )

        if output_path is None:
            return audio
        soundfile.write(output_path, audio, self.cfg.audio.sample_rate)





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