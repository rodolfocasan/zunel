# zunel/audio_analysis.py
import numpy as np
import parselmouth
import soundfile as sf
import pyloudnorm as pyln
from parselmouth.praat import call





REFERENCE_PITCH_HZ = 210.0
REFERENCE_ARTICULATION_RATE = 4.5
REFERENCE_LOUDNESS_LUFS = -20.0


def analyze_pitch(audio_path):
    sound = parselmouth.Sound(audio_path)
    
    pitch = sound.to_pitch_ac(
        time_step = None,
        pitch_floor = 75.0,
        max_number_of_candidates = 15,
        very_accurate = True,
        silence_threshold = 0.03,
        voicing_threshold = 0.45,
        octave_cost = 0.01,
        octave_jump_cost = 0.35,
        voiced_unvoiced_cost = 0.14,
        pitch_ceiling = 600.0
    )
    
    pitch_values = pitch.selected_array['frequency']
    pitch_strengths = pitch.selected_array['strength']
    
    valid_indices = (pitch_values > 0) & (pitch_strengths > 0.45)
    valid_pitches = pitch_values[valid_indices]
    valid_strengths = pitch_strengths[valid_indices]
    
    if len(valid_pitches) == 0:
        return 0
    
    weighted_pitch = np.average(valid_pitches, weights=valid_strengths)
    pitch_diff = weighted_pitch - REFERENCE_PITCH_HZ
    pitch_diff = np.clip(pitch_diff, -50, 50)
    return int(pitch_diff)


def analyze_speed(audio_path):
    sound = parselmouth.Sound(audio_path)
    
    intensity = call(sound, "To Intensity", 50, 0.0, "yes")
    textgrid = call(intensity, "To TextGrid (silences)", -25, 0.1, 0.05, "silent", "sounding")
    
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    n_segments = call(silencetable, "Get number of rows")
    
    phonation_time = 0
    for i in range(1, n_segments + 1):
        segment_start = call(silencetable, "Get value", i, 1)
        segment_end = call(silencetable, "Get value", i, 2)
        phonation_time += segment_end - segment_start
    
    if phonation_time == 0:
        return 0
    
    intensity_matrix = call(intensity, "Down to Matrix")
    sound_from_intensity = call(intensity_matrix, "To Sound (slice)", 1)
    point_process = call(sound_from_intensity, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    n_points = call(point_process, "Get number of points")
    
    pitch = call(sound, "To Pitch (ac)", 0.02, 30, 4, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
    
    n_syllables = 0
    for i in range(1, n_points + 1):
        time_point = call(point_process, "Get time from index", i)
        pitch_value = call(pitch, "Get value at time", time_point, "Hertz", "Linear")
        if pitch_value > 0:
            n_syllables += 1
    
    articulation_rate = n_syllables / phonation_time
    speed_ratio = (articulation_rate / REFERENCE_ARTICULATION_RATE) - 1.0
    speed_percent = speed_ratio * 100
    speed_percent = np.clip(speed_percent, -50, 100)
    return int(speed_percent)


def analyze_volume(audio_path):
    data, sr = sf.read(audio_path)
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(data)
    
    loudness_diff = loudness - REFERENCE_LOUDNESS_LUFS
    volume_percent = (loudness_diff / abs(REFERENCE_LOUDNESS_LUFS)) * 100
    volume_percent = np.clip(volume_percent, -50, 50)
    return int(volume_percent)


def analyze_audio(audio_path):
    try:
        pitch = analyze_pitch(audio_path)
    except Exception as e:
        print(f"[zunel] Warning: Could not analyze pitch: {e}")
        pitch = 0
    
    try:
        speed = analyze_speed(audio_path)
    except Exception as e:
        print(f"[zunel] Warning: Could not analyze speed: {e}")
        speed = 0
    
    try:
        volume = analyze_volume(audio_path)
    except Exception as e:
        print(f"[zunel] Warning: Could not analyze volume: {e}")
        volume = 0
    
    return {
        'pitch': pitch,
        'speed': speed,
        'volume': volume
    }