# zunel/text/text_processing.py
import re
import jieba
import cn2an
import inflect
import pykakasi
import eng_to_ipa as ipa
from jamo import h2j, j2hcj
from unidecode import unidecode
from pypinyin import lazy_pinyin, BOPOMOFO





_inflect_engine = inflect.engine()

_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

_abbreviations = [
    (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
        ('mrs', 'misess'), ('mr', 'mister'), ('dr', 'doctor'), ('st', 'saint'),
        ('co', 'company'), ('jr', 'junior'), ('maj', 'major'), ('gen', 'general'),
        ('drs', 'doctors'), ('rev', 'reverend'), ('lt', 'lieutenant'),
        ('hon', 'honorable'), ('sgt', 'sergeant'), ('capt', 'captain'),
        ('esq', 'esquire'), ('ltd', 'limited'), ('col', 'colonel'), ('ft', 'fort'),
    ]
]

_lazy_ipa_map = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('r', 'ɹ'), ('æ', 'e'), ('ɑ', 'a'), ('ɔ', 'o'), ('ð', 'z'), ('θ', 's'),
    ('ɛ', 'e'), ('ɪ', 'i'), ('ʊ', 'u'), ('ʒ', 'ʥ'), ('ʤ', 'ʥ'), ('ˈ', '↓'),
]]

_lazy_ipa2_map = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('r', 'ɹ'), ('ð', 'z'), ('θ', 's'), ('ʒ', 'ʑ'), ('ʤ', 'dʑ'), ('ˈ', '↓'),
]]

_ipa_to_ipa2_map = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('r', 'ɹ'), ('ʤ', 'dʒ'), ('ʧ', 'tʃ'),
]]

_latin_to_bopomofo = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', 'ㄟˉ'), ('b', 'ㄅㄧˋ'), ('c', 'ㄙㄧˉ'), ('d', 'ㄉㄧˋ'), ('e', 'ㄧˋ'),
    ('f', 'ㄝˊㄈㄨˋ'), ('g', 'ㄐㄧˋ'), ('h', 'ㄝˇㄑㄩˋ'), ('i', 'ㄞˋ'),
    ('j', 'ㄐㄟˋ'), ('k', 'ㄎㄟˋ'), ('l', 'ㄝˊㄛˋ'), ('m', 'ㄝˊㄇㄨˋ'),
    ('n', 'ㄣˉ'), ('o', 'ㄡˉ'), ('p', 'ㄆㄧˉ'), ('q', 'ㄎㄧㄡˉ'), ('r', 'ㄚˋ'),
    ('s', 'ㄝˊㄙˋ'), ('t', 'ㄊㄧˋ'), ('u', 'ㄧㄡˉ'), ('v', 'ㄨㄧˉ'),
    ('w', 'ㄉㄚˋㄅㄨˋㄌㄧㄡˋ'), ('x', 'ㄝˉㄎㄨˋㄙˋ'), ('y', 'ㄨㄞˋ'), ('z', 'ㄗㄟˋ'),
]]

_bopomofo_to_romaji = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'p⁼wo'), ('ㄆㄛ', 'pʰwo'), ('ㄇㄛ', 'mwo'), ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p⁼'), ('ㄆ', 'pʰ'), ('ㄇ', 'm'), ('ㄈ', 'f'), ('ㄉ', 't⁼'),
    ('ㄊ', 'tʰ'), ('ㄋ', 'n'), ('ㄌ', 'l'), ('ㄍ', 'k⁼'), ('ㄎ', 'kʰ'),
    ('ㄏ', 'h'), ('ㄐ', 'ʧ⁼'), ('ㄑ', 'ʧʰ'), ('ㄒ', 'ʃ'), ('ㄓ', 'ʦ`⁼'),
    ('ㄔ', 'ʦ`ʰ'), ('ㄕ', 's`'), ('ㄖ', 'ɹ`'), ('ㄗ', 'ʦ⁼'), ('ㄘ', 'ʦʰ'),
    ('ㄙ', 's'), ('ㄚ', 'a'), ('ㄛ', 'o'), ('ㄜ', 'ə'), ('ㄝ', 'e'),
    ('ㄞ', 'ai'), ('ㄟ', 'ei'), ('ㄠ', 'au'), ('ㄡ', 'ou'), ('ㄧㄢ', 'yeNN'),
    ('ㄢ', 'aNN'), ('ㄧㄣ', 'iNN'), ('ㄣ', 'əNN'), ('ㄤ', 'aNg'),
    ('ㄧㄥ', 'iNg'), ('ㄨㄥ', 'uNg'), ('ㄩㄥ', 'yuNg'), ('ㄥ', 'əNg'),
    ('ㄦ', 'əɻ'), ('ㄧ', 'i'), ('ㄨ', 'u'), ('ㄩ', 'ɥ'), ('ˉ', '→'),
    ('ˊ', '↑'), ('ˇ', '↓↑'), ('ˋ', '↓'), ('˙', ''), ('，', ','),
    ('。', '.'), ('！', '!'), ('？', '?'), ('—', '-'),
]]

_romaji_to_ipa = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('ʃy', 'ʃ'), ('ʧʰy', 'ʧʰ'), ('ʧ⁼y', 'ʧ⁼'), ('NN', 'n'), ('Ng', 'ŋ'),
    ('y', 'j'), ('h', 'x'),
]]

_bopomofo_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'p⁼wo'), ('ㄆㄛ', 'pʰwo'), ('ㄇㄛ', 'mwo'), ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p⁼'), ('ㄆ', 'pʰ'), ('ㄇ', 'm'), ('ㄈ', 'f'), ('ㄉ', 't⁼'),
    ('ㄊ', 'tʰ'), ('ㄋ', 'n'), ('ㄌ', 'l'), ('ㄍ', 'k⁼'), ('ㄎ', 'kʰ'),
    ('ㄏ', 'x'), ('ㄐ', 'tʃ⁼'), ('ㄑ', 'tʃʰ'), ('ㄒ', 'ʃ'), ('ㄓ', 'ts`⁼'),
    ('ㄔ', 'ts`ʰ'), ('ㄕ', 's`'), ('ㄖ', 'ɹ`'), ('ㄗ', 'ts⁼'), ('ㄘ', 'tsʰ'),
    ('ㄙ', 's'), ('ㄚ', 'a'), ('ㄛ', 'o'), ('ㄜ', 'ə'), ('ㄝ', 'ɛ'),
    ('ㄞ', 'aɪ'), ('ㄟ', 'eɪ'), ('ㄠ', 'ɑʊ'), ('ㄡ', 'oʊ'), ('ㄧㄢ', 'jɛn'),
    ('ㄩㄢ', 'ɥæn'), ('ㄢ', 'an'), ('ㄧㄣ', 'in'), ('ㄩㄣ', 'ɥn'), ('ㄣ', 'ən'),
    ('ㄤ', 'ɑŋ'), ('ㄧㄥ', 'iŋ'), ('ㄨㄥ', 'ʊŋ'), ('ㄩㄥ', 'jʊŋ'), ('ㄥ', 'əŋ'),
    ('ㄦ', 'əɻ'), ('ㄧ', 'i'), ('ㄨ', 'u'), ('ㄩ', 'ɥ'), ('ˉ', '→'),
    ('ˊ', '↑'), ('ˇ', '↓↑'), ('ˋ', '↓'), ('˙', ''), ('，', ','),
    ('。', '.'), ('！', '!'), ('？', '?'), ('—', '-'),
]]

_bopomofo_to_ipa2 = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'pwo'), ('ㄆㄛ', 'pʰwo'), ('ㄇㄛ', 'mwo'), ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p'), ('ㄆ', 'pʰ'), ('ㄇ', 'm'), ('ㄈ', 'f'), ('ㄉ', 't'),
    ('ㄊ', 'tʰ'), ('ㄋ', 'n'), ('ㄌ', 'l'), ('ㄍ', 'k'), ('ㄎ', 'kʰ'),
    ('ㄏ', 'h'), ('ㄐ', 'tɕ'), ('ㄑ', 'tɕʰ'), ('ㄒ', 'ɕ'), ('ㄓ', 'tʂ'),
    ('ㄔ', 'tʂʰ'), ('ㄕ', 'ʂ'), ('ㄖ', 'ɻ'), ('ㄗ', 'ts'), ('ㄘ', 'tsʰ'),
    ('ㄙ', 's'), ('ㄚ', 'a'), ('ㄛ', 'o'), ('ㄜ', 'ɤ'), ('ㄝ', 'ɛ'),
    ('ㄞ', 'aɪ'), ('ㄟ', 'eɪ'), ('ㄠ', 'ɑʊ'), ('ㄡ', 'oʊ'), ('ㄧㄢ', 'jɛn'),
    ('ㄩㄢ', 'yæn'), ('ㄢ', 'an'), ('ㄧㄣ', 'in'), ('ㄩㄣ', 'yn'), ('ㄣ', 'ən'),
    ('ㄤ', 'ɑŋ'), ('ㄧㄥ', 'iŋ'), ('ㄨㄥ', 'ʊŋ'), ('ㄩㄥ', 'jʊŋ'), ('ㄥ', 'ɤŋ'),
    ('ㄦ', 'əɻ'), ('ㄧ', 'i'), ('ㄨ', 'u'), ('ㄩ', 'y'), ('ˉ', '˥'),
    ('ˊ', '˧˥'), ('ˇ', '˨˩˦'), ('ˋ', '˥˩'), ('˙', ''), ('，', ','),
    ('。', '.'), ('！', '!'), ('？', '?'), ('—', '-'),
]]

_pad = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '

symbols = [_pad] + list(_punctuation) + list(_letters)

SPACE_ID = symbols.index(" ")

num_zh_tones = 6
num_ja_tones = 1
num_en_tones = 4
num_kr_tones = 1

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
    "KR": num_zh_tones + num_ja_tones + num_en_tones,
}

_sym_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_sym = {i: s for i, s in enumerate(symbols)}





def encode_text(text, sym_list, cleaner_names):
    sym_map = {}
    i = 0
    for s in sym_list:
        sym_map[s] = i
        i = i + 1

    cleaned = _apply_cleaners(text, cleaner_names)
    print(cleaned)
    print(" - length:" + str(len(cleaned)))

    seq = []
    for ch in cleaned:
        if ch not in sym_map:
            continue
        seq.append(sym_map[ch])
    print(" - length:" + str(len(seq)))
    return seq


def encode_cleaned_text(cleaned_text, sym_list):
    sym_map = {s: i for i, s in enumerate(sym_list)}
    return [sym_map[ch] for ch in cleaned_text if ch in sym_map]


def encode_text_multilingual(cleaned_text, tones, language, sym_list, lang_list):
    sym_map = {s: i for i, s in enumerate(sym_list)}
    lang_map = {s: i for i, s in enumerate(lang_list)}
    phones = [sym_map[ch] for ch in cleaned_text]
    tone_offset = language_tone_start_map[language]
    tones = [t + tone_offset for t in tones]
    lang_id = lang_map[language]
    lang_ids = [lang_id for _ in phones]
    return phones, tones, lang_ids


def decode_sequence(sequence):
    return ''.join(_id_to_sym[sid] for sid in sequence)


def _apply_cleaners(text, cleaner_names):
    for name in cleaner_names:
        cleaner = _CLEANER_MAP.get(name)
        if not cleaner:
            raise Exception('[zunel] Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def multilingual_cleaner(text):
    text = re.sub(r'\[ZH\](.*?)\[ZH\]', lambda x: zh_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean_to_ipa(x.group(1)) + ' ', text)
    text = re.sub(r'\[EN\](.*?)\[EN\]', lambda x: en_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text


_CLEANER_MAP = {
    'cjke_cleaners2': multilingual_cleaner,
}


def expand_abbrev(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text)


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    
    if len(parts) > 2:
        return match + ' dollars'
    
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        return '%s %s, %s %s' % (
            dollars, 'dollar' if dollars == 1 else 'dollars',
            cents, 'cent' if cents == 1 else 'cents',
        )
    elif dollars:
        return '%s %s' % (dollars, 'dollar' if dollars == 1 else 'dollars')
    elif cents:
        return '%s %s' % (cents, 'cent' if cents == 1 else 'cents')
    return 'zero dollars'


def _expand_ordinal(m):
    return _inflect_engine.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return 'two thousand'
        elif 2000 < num < 2010:
            return 'two thousand ' + _inflect_engine.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect_engine.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect_engine.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    return _inflect_engine.number_to_words(num, andword='')


def expand_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_re, _expand_decimal, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def _mark_dark_l(text):
    return re.sub(r'l([^aeiouæɑɔəɛɪʊ ]*(?: |$))', lambda x: 'ɫ' + x.group(1), text)


def en_to_ipa(text):
    text = unidecode(text).lower()
    text = expand_abbrev(text)
    text = expand_numbers(text)
    phonemes = ipa.convert(text)
    return normalize_whitespace(phonemes)


def en_to_lazy_ipa(text):
    text = en_to_ipa(text)
    for regex, replacement in _lazy_ipa_map:
        text = re.sub(regex, replacement, text)
    return text


def en_to_ipa2(text):
    text = en_to_ipa(text)
    text = _mark_dark_l(text)
    for regex, replacement in _ipa_to_ipa2_map:
        text = re.sub(regex, replacement, text)
    return text.replace('...', '…')


def en_to_lazy_ipa2(text):
    text = en_to_ipa(text)
    for regex, replacement in _lazy_ipa2_map:
        text = re.sub(regex, replacement, text)
    return text


def number_to_zh(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for n in numbers:
        text = text.replace(n, cn2an.an2cn(n), 1)
    return text


def zh_to_bopomofo(text):
    text = text.replace('、', '，').replace('；', '，').replace('：', '，')
    words = jieba.lcut(text, cut_all=False)
    result = ''
    for word in words:
        bopo = lazy_pinyin(word, BOPOMOFO)
        
        if not re.search('[\u4e00-\u9fff]', word):
            result += word
            continue
        
        for i in range(len(bopo)):
            bopo[i] = re.sub(r'([\u3105-\u3129])$', r'\1ˉ', bopo[i])
        
        if result:
            result += ' '
        result += ''.join(bopo)
    return result


def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text


def _bopomofo_to_romaji_str(text):
    for regex, replacement in _bopomofo_to_romaji:
        text = re.sub(regex, replacement, text)
    return text


def _bopomofo_to_ipa_str(text):
    for regex, replacement in _bopomofo_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


def _bopomofo_to_ipa2_str(text):
    for regex, replacement in _bopomofo_to_ipa2:
        text = re.sub(regex, replacement, text)
    return text


def zh_to_romaji(text):
    text = number_to_zh(text)
    text = zh_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = _bopomofo_to_romaji_str(text)
    text = re.sub('i([aoe])', r'y\1', text)
    text = re.sub('u([aoəe])', r'w\1', text)
    text = re.sub('([ʦsɹ]`[⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
    text = re.sub('([ʦs][⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
    return text


def zh_to_lazy_ipa(text):
    text = zh_to_romaji(text)
    for regex, replacement in _romaji_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


def zh_to_ipa(text):
    text = number_to_zh(text)
    text = zh_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = _bopomofo_to_ipa_str(text)
    text = re.sub('i([aoe])', r'j\1', text)
    text = re.sub('u([aoəe])', r'w\1', text)
    text = re.sub('([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
    text = re.sub('([s][⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
    return text


def zh_to_ipa2(text):
    text = number_to_zh(text)
    text = zh_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = _bopomofo_to_ipa2_str(text)
    text = re.sub(r'i([aoe])', r'j\1', text)
    text = re.sub(r'u([aoəe])', r'w\1', text)
    text = re.sub(r'([ʂɹ]ʰ?)([˩˨˧˦˥ ]+|$)', r'\1ʅ\2', text)
    text = re.sub(r'(sʰ?)([˩˨˧˦˥ ]+|$)', r'\1ɿ\2', text)
    return text


def japanese_to_ipa2(text):
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    return ' '.join(item['hepburn'] for item in result)


def korean_to_ipa(text):
    return j2hcj(h2j(text))