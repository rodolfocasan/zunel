# zunel/voice_config.py





VOICE_REGISTRY = {
    'es-latino': {
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
    'es-spain': {
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





CALIBRATION_TEXTS = {
    'es-latino': [
        'El veloz murciélago hindú comía feliz cardillo y kiwi.',
        'El resplandor del sol acaricia las olas del mar.',
        'La vida es un camino lleno de sorpresas y aprendizajes.',
        'Hoy el cielo está despejado y el viento sopla suavemente.'
    ],
    'es-spain': [
        'El veloz murciélago hindú comía feliz cardillo y kiwi.',
        'El resplandor del sol acaricia las olas del mar.',
        'La vida es un camino lleno de sorpresas y aprendizajes.',
        'Hoy el cielo está despejado y el viento sopla suavemente.'
    ],
    'en': [
        'The quick brown fox jumps over the lazy dog.',
        'She sells seashells by the seashore.',
        'How much wood would a woodchuck chuck if a woodchuck could chuck wood?',
        'Peter Piper picked a peck of pickled peppers.'
    ],
    'fr': [
        'Portez ce vieux whisky au juge blond qui fume.',
        'La lueur dorée du soleil caresse les vagues.',
        'Le vent souffle doucement sur les collines verdoyantes.',
        'Chaque jour est une nouvelle occasion de grandir.'
    ],
    'de': [
        'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern.',
        'Die Sonne strahlt hell am klaren Himmel.',
        'Der Wind weht sanft über die grünen Hügel.',
        'Jeder Tag bringt neue Möglichkeiten und Chancen.'
    ],
    'pt': [
        'O veloz murciélago hindu comia feliz cardillo e kiwi.',
        'A luz do sol acaricia as ondas do mar.',
        'A vida é um caminho cheio de surpresas e aprendizados.',
        'Hoje o céu está limpo e o vento sopra suavemente.'
    ],
    'ru': [
        'Съешь ещё этих мягких французских булок да выпей чаю.',
        'Солнце ярко светит в ясном небе.',
        'Ветер мягко дует над зелёными холмами.',
        'Каждый день приносит новые возможности.'
    ],
    'ja': [
        '彼は毎朝ジョギングをして体を健康に保っています。',
        '今日はとても良い天気で散歩日和です。',
        '日本の春は桜の花が美しく咲き誇ります。',
        '毎日少しずつ努力することが大切です。'
    ],
    'ko': [
        '안녕하세요! 오늘은 날씨가 정말 좋네요.',
        '한국의 봄은 벚꽃이 아름답게 피어납니다.',
        '매일 조금씩 노력하는 것이 중요합니다.',
        '오늘도 좋은 하루 되세요.'
    ],
    'zh': [
        '能大声说出来的秘密才是真正的秘密。',
        '在这次旅行中我们欣赏了很多美丽的风景。',
        '学习是一件让人快乐的事情。',
        '今天天气晴朗微风吹拂令人心旷神怡。'
    ]
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


def get_calibration_texts(language):
    if language not in CALIBRATION_TEXTS:
        raise ValueError(f"[zunel] Calibration texts not available for language '{language}'")
    return CALIBRATION_TEXTS[language]