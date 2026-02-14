# zunel/voice_encoder.py
import os
import base64
import hashlib
import librosa
import numpy as np
from glob import glob
from pydub import AudioSegment
from faster_whisper import WhisperModel
from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments

import torch





_asr_model_size = "medium"
_asr_model = None


def segment_with_asr(audio_path, audio_name, target_dir='processed'):
    global _asr_model

    if _asr_model is None:
        _asr_model = WhisperModel(
            _asr_model_size,
            device = "cuda",
            compute_type = "float16"
        )

    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)

    out_dir = os.path.join(target_dir, audio_name)
    wavs_dir = os.path.join(out_dir, 'wavs')

    if not os.path.exists(wavs_dir):
        os.makedirs(wavs_dir, exist_ok=True)

    segments, info = _asr_model.transcribe(
        audio_path,
        beam_size = 5,
        word_timestamps = True
    )

    segments = list(segments)
    seg_idx = 0
    start_time = None

    k = 0
    while k < len(segments):
        seg = segments[k]

        if k == 0:
            start_time = max(0, seg.start)

        end_time = seg.end

        if seg.words:
            prob_sum = 0.0
            word_count = 0

            for w in seg.words:
                prob_sum = prob_sum + w.probability
                word_count = word_count + 1
            confidence = prob_sum / word_count
        else:
            confidence = 0.0

        text = seg.text.replace('...', '')

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000) + 80
        end_ms = min(max_len, end_ms)

        chunk = audio[start_ms:end_ms]
        dur = chunk.duration_seconds

        cond1 = dur > 1.5
        cond2 = dur < 20.0
        cond3 = len(text) >= 2
        cond4 = len(text) < 200

        if cond1 and cond2 and cond3 and cond4:
            out_path = os.path.join(wavs_dir, audio_name + "_seg" + str(seg_idx) + ".wav")
            chunk.export(out_path, format='wav')

        if k < len(segments) - 1:
            next_start = segments[k + 1].start - 0.08
            start_time = max(0, next_start)

        seg_idx = seg_idx + 1
        k = k + 1
    return wavs_dir



def segment_with_vad(audio_path, audio_name, target_dir, split_seconds=10.0):
    SAMPLE_RATE = 16000

    raw = get_audio_tensor(audio_path)

    vad_segs_raw = get_vad_segments(
        raw,
        output_sample = True,
        min_speech_duration = 0.1,
        min_silence_duration = 1,
        method = "silero"
    )

    vad_segs = []
    for s in vad_segs_raw:
        start = float(s["start"]) / SAMPLE_RATE
        end = float(s["end"]) / SAMPLE_RATE
        vad_segs.append((start, end))
    print(vad_segs)

    audio = AudioSegment.from_file(audio_path)
    active = AudioSegment.silent(duration=0)

    for pair in vad_segs:
        t0 = pair[0]
        t1 = pair[1]

        start_ms = int(t0 * 1000)
        end_ms = int(t1 * 1000)
        active = active + audio[start_ms:end_ms]
    dur = active.duration_seconds
    print("[zunel] after vad: dur = " + str(dur))

    out_dir = os.path.join(target_dir, audio_name)
    wavs_dir = os.path.join(out_dir, 'wavs')
    if not os.path.exists(wavs_dir):
        os.makedirs(wavs_dir, exist_ok=True)

    n_splits = int(np.round(dur / split_seconds))
    assert n_splits > 0, 'input audio is too short'

    interval = dur / n_splits
    t0 = 0.0
    count = 0

    i = 0
    while i < n_splits:
        if i == n_splits - 1:
            t1 = dur
        else:
            t1_candidate = t0 + interval
            t1 = min(t1_candidate, dur)

        start_ms = int(t0 * 1000)
        end_ms = int(t1 * 1000)
        chunk = active[start_ms:end_ms]

        out_path = os.path.join(wavs_dir, audio_name + "_seg" + str(count) + ".wav")
        chunk.export(out_path, format='wav')

        t0 = t1
        count = count + 1
        i = i + 1
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
        wavs_dir = segment_with_vad(audio_path, target_dir=target_dir, audio_name=audio_name)
    else:
        wavs_dir = segment_with_asr(audio_path, target_dir=target_dir, audio_name=audio_name)

    wav_files = glob(f'{wavs_dir}/*.wav')
    if not wav_files:
        raise NotImplementedError('[zunel] No audio segments found!')
    return vc_model.extract_se(wav_files, se_save_path=se_path), audio_name