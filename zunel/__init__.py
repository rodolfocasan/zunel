# zunel/__init__.py
from zunel.engine import TimbreConverter, VoiceCloner
from zunel.audio_analysis import analyze_audio, analyze_pitch, analyze_speed, analyze_volume
from zunel.voice_config import get_voice, get_calibration_texts, VOICE_REGISTRY
from zunel.speaker_encoder import ImprovedSpeakerEmbedder, MultiSampleEmbeddingAggregator
from zunel.perturbation import AudioPerturbation, extract_robust_embedding_multi_sample





__version__ = '1.0.2'
__all__ = [
    'TimbreConverter',
    'VoiceCloner',
    'analyze_audio',
    'analyze_pitch',
    'analyze_speed',
    'analyze_volume',
    'get_voice',
    'get_calibration_texts',
    'VOICE_REGISTRY',
    'ImprovedSpeakerEmbedder',
    'MultiSampleEmbeddingAggregator',
    'AudioPerturbation',
    'extract_robust_embedding_multi_sample'
]