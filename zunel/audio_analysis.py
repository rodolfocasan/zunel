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


def is_extreme_voice(pitch_hz, detected_gender):
    if detected_gender == 'male':
        return pitch_hz < 100 or pitch_hz > 140
    else:
        return pitch_hz < 190 or pitch_hz > 230


def analyze_formants(audio_path):
    sound = parselmouth.Sound(audio_path)
    
    formant = sound.to_formant_burg(
        time_step = 0.01,
        max_number_of_formants = 5,
        maximum_formant = 5500.0,
        window_length = 0.025,
        pre_emphasis_from = 50.0
    )
    
    f1_values = []
    f2_values = []
    f3_values = []
    n_frames = call(formant, "Get number of frames")
    for i in range(1, n_frames + 1):
        f1 = call(formant, "Get value at time", 1, formant.get_time_from_frame_number(i), "hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, formant.get_time_from_frame_number(i), "hertz", "Linear")
        f3 = call(formant, "Get value at time", 3, formant.get_time_from_frame_number(i), "hertz", "Linear")
        
        if not np.isnan(f1) and f1 > 0:
            f1_values.append(f1)
        if not np.isnan(f2) and f2 > 0:
            f2_values.append(f2)
        if not np.isnan(f3) and f3 > 0:
            f3_values.append(f3)
    
    formant_data = {
        'f1_mean': np.mean(f1_values) if f1_values else 0,
        'f2_mean': np.mean(f2_values) if f2_values else 0,
        'f3_mean': np.mean(f3_values) if f3_values else 0,
        'formant_dispersion': 0
    }
    
    if f1_values and f2_values and f3_values:
        formant_dispersion = np.sqrt(
            (formant_data['f1_mean'] / 1.0) ** 2 +
            (formant_data['f2_mean'] / 2.0) ** 2 +
            (formant_data['f3_mean'] / 3.0) ** 2
        )
        formant_data['formant_dispersion'] = formant_dispersion
    return formant_data


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
        return 0, None, False
    
    weighted_pitch = np.average(valid_pitches, weights=valid_strengths)
    detected_gender = detect_gender_from_pitch(weighted_pitch)
    is_extreme = is_extreme_voice(weighted_pitch, detected_gender)
    return int(weighted_pitch), detected_gender, is_extreme


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
        return articulation_rate
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
        return loudness
    except Exception as e:
        print(f"[zunel] Volume analysis error: {e}")
        return REFERENCE_LOUDNESS_LUFS


def analyze_audio(audio_path):
    pitch_hz, detected_gender, is_extreme = analyze_pitch(audio_path)
    speech_rate = analyze_speed(audio_path)
    loudness_lufs = analyze_volume(audio_path)
    formant_data = analyze_formants(audio_path)
    
    return {
        'pitch_hz': pitch_hz,
        'detected_gender': detected_gender,
        'is_extreme_voice': is_extreme,
        'speech_rate_sps': speech_rate,
        'loudness_lufs': loudness_lufs,
        'formant_f1_mean': formant_data['f1_mean'],
        'formant_f2_mean': formant_data['f2_mean'],
        'formant_f3_mean': formant_data['f3_mean'],
        'formant_dispersion': formant_data['formant_dispersion']
    }