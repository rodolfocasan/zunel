# zunel/voices/voice_encoder.py
import os
import base64
import hashlib
import librosa
import numpy as np
from glob import glob
from pydub import AudioSegment

import torch





_silero_model = None
_silero_get_ts = None
_silero_read_audio = None


def _load_silero():
    global _silero_model, _silero_get_ts, _silero_read_audio
    
    if _silero_model is not None:
        return _silero_model, _silero_get_ts, _silero_read_audio

    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
        _silero_model = load_silero_vad()
        _silero_get_ts = get_speech_timestamps
        _silero_read_audio = read_audio
    except ImportError:
        model, utils = torch.hub.load(
            'snakers4/silero-vad',
            'silero_vad',
            trust_repo = True,
            verbose = False
        )
        _silero_model = model
        _silero_get_ts = utils[0]
        _silero_read_audio = utils[2]
    return _silero_model, _silero_get_ts, _silero_read_audio


def segment_with_asr(audio_path, audio_name, target_dir='processed'):
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    vad_model, get_speech_timestamps, read_audio = _load_silero()

    SAMPLE_RATE = 16000
    wav = read_audio(audio_path, sampling_rate=SAMPLE_RATE)
    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate = SAMPLE_RATE,
        min_speech_duration_ms = 1500,
        max_speech_duration_s = 20.0,
        min_silence_duration_ms = 300,
        return_seconds = True
    )

    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)

    out_dir = os.path.join(target_dir, audio_name)
    wavs_dir = os.path.join(out_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)

    seg_idx = 0
    for ts in timestamps:
        start_ms = int(ts['start'] * 1000)
        end_ms = min(int(ts['end'] * 1000) + 80, max_len)
        chunk = audio[start_ms:end_ms]
        dur = chunk.duration_seconds

        if not (1.5 <= dur <= 20.0):
            continue

        out_path = os.path.join(wavs_dir, f'{audio_name}_seg{seg_idx}.wav')

        try:
            tmp_path = out_path + '._tmp.wav'
            chunk.set_frame_rate(16000).set_channels(1).export(tmp_path, format='wav')
            
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
            
            text = recognizer.recognize_google(audio_data)
            
            os.remove(tmp_path)
            if len(text.strip()) >= 2:
                chunk.export(out_path, format='wav')
                seg_idx += 1
        except sr.UnknownValueError:
            if os.path.exists(out_path + '._tmp.wav'):
                os.remove(out_path + '._tmp.wav')
        except Exception:
            if os.path.exists(out_path + '._tmp.wav'):
                try:
                    os.remove(out_path + '._tmp.wav')
                except OSError:
                    pass
            chunk.export(out_path, format='wav')
            seg_idx += 1
    return wavs_dir


def segment_with_vad(audio_path, audio_name, target_dir, split_seconds=10.0):
    SAMPLE_RATE = 16000

    vad_model, get_speech_timestamps, read_audio = _load_silero()
    wav = read_audio(audio_path, sampling_rate=SAMPLE_RATE)

    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate = SAMPLE_RATE,
        min_speech_duration_ms = 100,
        min_silence_duration_ms = 1000,
        return_seconds = False
    )

    audio_full = AudioSegment.from_file(audio_path)
    active = AudioSegment.silent(duration=0)

    for seg in timestamps:
        start_ms = int(seg['start'] / SAMPLE_RATE * 1000)
        end_ms = int(seg['end'] / SAMPLE_RATE * 1000)
        active = active + audio_full[start_ms:end_ms]

    dur = active.duration_seconds
    print(f"[zunel] after vad: dur = {dur}")

    out_dir = os.path.join(target_dir, audio_name)
    wavs_dir = os.path.join(out_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)

    n_splits = max(1, int(np.round(dur / split_seconds)))
    interval = dur / n_splits
    t0 = 0.0
    count = 0

    for i in range(n_splits):
        t1 = dur if i == n_splits - 1 else min(t0 + interval, dur)
        start_ms = int(t0 * 1000)
        end_ms = int(t1 * 1000)
        chunk = active[start_ms:end_ms]
        out_path = os.path.join(wavs_dir, f'{audio_name}_seg{count}.wav')
        chunk.export(out_path, format='wav')
        t0 = t1
        count += 1
    return wavs_dir


def audio_fingerprint(audio_path):
    arr, _ = librosa.load(audio_path, sr=None, mono=True)
    h = hashlib.sha256(arr.tobytes()).digest()
    return base64.b64encode(h).decode('utf-8')[:16].replace('/', '_^')


def extract_speaker_embedding(audio_path, vc_model, target_dir='processed', vad=True):
    version = vc_model.version
    print("[zunel] version:", version)

    audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{version}_{audio_fingerprint(audio_path)}"
    se_path = os.path.join(target_dir, audio_name, 'se.pth')

    if vad:
        wavs_dir = segment_with_vad(audio_path, target_dir=target_dir, audio_name=audio_name)
    else:
        wavs_dir = segment_with_asr(audio_path, target_dir=target_dir, audio_name=audio_name)

    wav_files = glob(f'{wavs_dir}/*.wav')
    
    if not wav_files:
        raise NotImplementedError('[zunel] No audio segments found!')
    
    return vc_model.extract_se(wav_files, se_save_path=se_path), audio_name