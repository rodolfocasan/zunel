# zunel/voice_encoder.py
import os
import base64
import hashlib
import librosa
import numpy as np
from glob import glob
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import torch





def segment_with_vad(audio_path, audio_name, target_dir, split_seconds=10.0):
    audio = AudioSegment.from_file(audio_path)

    db_floor = audio.dBFS if audio.dBFS != float('-inf') else -40.0
    silence_thresh = db_floor - 14

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len = 300,
        silence_thresh = silence_thresh,
        seek_step = 10
    )

    if not nonsilent_ranges:
        nonsilent_ranges = [(0, len(audio))]

    active = AudioSegment.silent(duration=0)
    for start_ms, end_ms in nonsilent_ranges:
        active += audio[start_ms:end_ms]

    dur = active.duration_seconds
    if dur == 0.0:
        dur = audio.duration_seconds
        active = audio

    out_dir = os.path.join(target_dir, audio_name)
    wavs_dir = os.path.join(out_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)

    n_splits = max(1, int(np.round(dur / split_seconds)))
    interval = dur / n_splits

    count = 0
    t0 = 0.0

    for i in range(n_splits):
        t1 = dur if i == n_splits - 1 else min(t0 + interval, dur)
        chunk = active[int(t0 * 1000):int(t1 * 1000)]
        out_path = os.path.join(wavs_dir, f"{audio_name}_seg{count}.wav")
        chunk.export(out_path, format='wav')
        t0 = t1
        count += 1
    return wavs_dir


def segment_with_asr(audio_path, audio_name, target_dir='processed'):
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    audio_pydub = AudioSegment.from_file(audio_path)

    db_floor = audio_pydub.dBFS if audio_pydub.dBFS != float('-inf') else -40.0
    silence_thresh = db_floor - 14

    nonsilent_ranges = detect_nonsilent(
        audio_pydub,
        min_silence_len = 500,
        silence_thresh = silence_thresh,
        seek_step = 10
    )

    if not nonsilent_ranges:
        nonsilent_ranges = [(0, len(audio_pydub))]

    out_dir = os.path.join(target_dir, audio_name)
    wavs_dir = os.path.join(out_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)

    seg_idx = 0
    for start_ms, end_ms in nonsilent_ranges:
        chunk = audio_pydub[start_ms:end_ms]
        dur = chunk.duration_seconds

        if dur < 1.5 or dur > 20.0:
            continue

        text = ''
        try:
            chunk_16k = chunk.set_frame_rate(16000).set_channels(1)
            tmp_path = os.path.join(wavs_dir, f"_tmp_{seg_idx}.wav")
            chunk_16k.export(tmp_path, format='wav')

            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)

            text = recognizer.recognize_google(audio_data)

            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        if len(text) > 200:
            continue

        out_path = os.path.join(wavs_dir, f"{audio_name}_seg{seg_idx}.wav")
        chunk.export(out_path, format='wav')
        seg_idx += 1
    return wavs_dir


def audio_fingerprint(audio_path):
    arr, _ = librosa.load(audio_path, sr=None, mono=True)
    h = hashlib.sha256(arr.tobytes()).digest()
    return base64.b64encode(h).decode('utf-8')[:16].replace('/', '_^')


def extract_speaker_embedding(audio_path, vc_model, target_dir='processed', vad=True):
    device = vc_model.device
    version = vc_model.version
    print("Zunel version:", version)

    audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{version}_{audio_fingerprint(audio_path)}"
    se_path = os.path.join(target_dir, audio_name, 'se.pth')

    if vad:
        wavs_dir = segment_with_vad(audio_path, audio_name=audio_name, target_dir=target_dir)
    else:
        wavs_dir = segment_with_asr(audio_path, audio_name=audio_name, target_dir=target_dir)

    wav_files = glob(f'{wavs_dir}/*.wav')
    if not wav_files:
        raise NotImplementedError('[zunel] No audio segments found!')
    return vc_model.extract_se(wav_files, se_save_path=se_path), audio_name