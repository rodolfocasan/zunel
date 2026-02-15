# zunel/__init__.py
from zunel.engine import TimbreConverter, VoiceCloner, normalize_language_code
from zunel.audio_analysis import analyze_audio, analyze_pitch, analyze_speed, analyze_volume
from zunel.voice_config import get_voice, get_calibration_texts, VOICE_REGISTRY





__version__ = '1.0.2'
__all__ = [
    'TimbreConverter',
    'VoiceCloner',
    'normalize_language_code',
    'analyze_audio',
    'analyze_pitch',
    'analyze_speed',
    'analyze_volume',
    'get_voice',
    'get_calibration_texts',
    'VOICE_REGISTRY'
]