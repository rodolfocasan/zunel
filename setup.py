# setup.py
from setuptools import setup, find_packages


setup(
    name = "zunel",
    version = "1.0.2",
    packages = find_packages(),
    install_requires = [
        "torch>=2.0.0",
        "numpy",
        "soundfile",
        "librosa",
        "pydub",
        "faster-whisper",
        "whisper-timestamped",
        "langid",
        "gradio",
        "inflect",
        "unidecode",
        "eng-to-ipa",
        "pypinyin",
        "jieba",
        "cn2an",
        "pykakasi",
        "jamo",
        "praat-parselmouth",
        "pyloudnorm",
        "pyworld"
    ],
    python_requires = ">=3.8",
)