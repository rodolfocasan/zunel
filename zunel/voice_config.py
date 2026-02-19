# zunel/voice_config.py





VOICE_REGISTRY = {
    'es-latam': {
        'male': [
            'es-CL-LorenzoNeural',
            'es-MX-JorgeNeural'
        ],
        'female': [
            'es-CL-CatalinaNeural',
            'es-UY-ValentinaNeural',
            'es-EC-AndreaNeural'
        ]
    },
    'es-es': {
        'male': [
            'es-ES-AlvaroNeural'
        ],
        'female': [
            'es-CU-BelkysNeural',
            'es-ES-ElviraNeural',
            'es-ES-XimenaNeural'
        ]
    },
    'en': {
        'male': [
            'en-US-AndrewMultilingualNeural',
            'en-HK-SamNeural'
        ],
        'female': [
            'en-US-AriaNeural',
            'en-AU-NatashaNeural'
        ]
    },
    'fr': {
        'male': [
            'fr-CA-ThierryNeural',
            'fr-FR-RemyMultilingualNeural'
        ],
        'female': [
            'fr-FR-VivienneMultilingualNeural'
        ]
    },
    'de': {
        'male': [
            'de-CH-JanNeural',
            'de-DE-ConradNeural'
        ],
        'female': [
            'de-CH-LeniNeural',
            'de-DE-KatjaNeural'
        ]
    },
    'pt': {
        'male': [
            'pt-BR-AntonioNeural'
        ],
        'female': [
            'pt-BR-ThalitaMultilingualNeural',
            'pt-BR-FranciscaNeural'
        ]
    },
    'ru': {
        'male': [
            'ru-RU-DmitryNeural'
        ],
        'female': [
            'ru-RU-SvetlanaNeural'
        ]
    },
    'ja': {
        'male': [
            'ja-JP-KeitaNeural'
        ],
        'female': [
            'ja-JP-NanamiNeural'
        ]
    },
    'ko': {
        'male': [
            'ko-KR-InJoonNeural'
        ],
        'female': [
            'ko-KR-SunHiNeural'
        ]
    },
    'zh': {
        'male': [
            'zh-TW-YunJheNeural',
            'zh-CN-YunjianNeural'
        ],
        'female': [
            'zh-HK-HiuMaanNeural',
            'zh-CN-shaanxi-XiaoniNeural'
        ]
    }
}


def get_voice(language, gender, version=0):
    if language not in VOICE_REGISTRY:
        raise ValueError(f"[zunel] Language '{language}' not supported. Available: {list(VOICE_REGISTRY.keys())}")

    if gender not in ['male', 'female']:
        raise ValueError(f"[zunel] Gender must be 'male' or 'female', got '{gender}'")

    voices = VOICE_REGISTRY[language][gender]

    if version >= len(voices):
        raise ValueError(f"[zunel] Version {version} not available for {language}/{gender}. Max version: {len(voices) - 1}")
    return voices[version]