# zunel/audio_analysis.py
import numpy as np
import parselmouth
import soundfile as sf
import pyloudnorm as pyln
from parselmouth.praat import call





MALE_AVG_PITCH_HZ = 120.0
FEMALE_AVG_PITCH_HZ = 210.0
GENDER_THRESHOLD_HZ = 165.0

NORMAL_SPEECH_RATE_SPS = 4.5

REFERENCE_LOUDNESS_LUFS = -20.0


def detect_gender_from_pitch(mean_pitch_hz):
    return 'male' if mean_pitch_hz < GENDER_THRESHOLD_HZ else 'female'


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
        return 0, None
    
    weighted_pitch = np.average(valid_pitches, weights=valid_strengths)
    detected_gender = detect_gender_from_pitch(weighted_pitch)
    
    reference_pitch = MALE_AVG_PITCH_HZ if detected_gender == 'male' else FEMALE_AVG_PITCH_HZ
    
    pitch_deviation = weighted_pitch - reference_pitch
    pitch_deviation = np.clip(pitch_deviation, -50, 50)
    return int(pitch_deviation), detected_gender


def analyze_speed(audio_path):
    sound = parselmouth.Sound(audio_path)
    
    try:
        intensity = call(sound, "To Intensity", 50, 0.0, "yes")
        textgrid = call(intensity, "To TextGrid (silences)", -25, 0.1, 0.05, "silent", "sounding")
        
        silencetier = call(textgrid, "Extract tier", 1)
        silencetable = call(silencetier, "Down to TableOfReal", "sounding")
        n_segments = call(silencetable, "Get number of rows")
        
        phonation_time = 0.0
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
        
        pitch_obj = call(sound, "To Pitch (ac)", 0.02, 30, 4, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
        
        n_syllables = 0
        for i in range(1, n_points + 1):
            time_point = call(point_process, "Get time from index", i)
            pitch_value = call(pitch_obj, "Get value at time", time_point, "Hertz", "Linear")
            if pitch_value > 0:
                n_syllables += 1
        
        if n_syllables == 0:
            return 0
        
        articulation_rate = n_syllables / phonation_time
        
        rate_deviation = articulation_rate - NORMAL_SPEECH_RATE_SPS
        speed_percent = (rate_deviation / NORMAL_SPEECH_RATE_SPS) * 100
        speed_percent = np.clip(speed_percent, -50, 100)
        return int(speed_percent)
    except Exception as e:
        print(f"[zunel] Speed analysis error: {e}")
        return 0


def analyze_volume(audio_path):
    try:
        data, sr = sf.read(audio_path)
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(data)
        
        loudness_deviation = loudness - REFERENCE_LOUDNESS_LUFS
        volume_percent = (loudness_deviation / abs(REFERENCE_LOUDNESS_LUFS)) * 100
        volume_percent = np.clip(volume_percent, -50, 50)
        return int(volume_percent)
    except Exception as e:
        print(f"[zunel] Volume analysis error: {e}")
        return 0


def analyze_audio(audio_path):
    pitch, detected_gender = analyze_pitch(audio_path)
    speed = analyze_speed(audio_path)
    volume = analyze_volume(audio_path)
    
    return {
        'pitch': pitch,
        'speed': speed,
        'volume': volume,
        'detected_gender': detected_gender
    }