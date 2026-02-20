# zunel - Made with ❤️ in El Salvador
Multilingual voice cloning and timbre conversion engine. Given a reference audio clip, zunel transfers the speaker's timbre to synthesized speech in any supported language while preserving prosody and naturalness.





## Supported Languages
| Language | Code |
|---|---|
| Latam Spanish | `es-latam` |
| Spain Spanish | `es` |
| English | `en` |
| French | `fr` |
| German | `de` |
| Portuguese | `pt` |
| Russian | `ru` |
| Japanese | `ja` |
| Korean | `ko` |
| Mandarin Chinese | `zh` |





## Installation
```bash
pip3 install --upgrade git+https://github.com/rodolfocasan/zunel.git
```

Python 3.8+ and PyTorch 2.0+ are required.





# Quick usage
## Inference saving output
```python
import torch
import asyncio
from cencalang_tts import TTSGenerator

from zunel import TimbreConverter, VoiceCloner, download_models



adapter_path, model_path, model_config_path = download_models()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

converter = TimbreConverter(model_config_path, device=device)
converter.load_ckpt(model_path)
converter.load_adapters(adapter_path)

if device == 'cpu':
    converter.optimize_for_cpu(
        quantize = True,
        compile_model = False,
        thread_mode = 'deterministic' # or 'max_speed'
    )

tts = TTSGenerator()
cloner = VoiceCloner(converter, tts)

async def main():
    await cloner.clone_voice(
        reference_audio_path = 'my_voice_in_any_language.wav', # spanish, english, etc
        target_language = 'pt', # language you want to translate your voice to
        target_text = "Tecnicamente é possível. Mas levaria mais tempo, pois eu precisaria ler a documentação e alterar toda a arquitetura.", # input text (in language you want to translate your voice to)
        gender = 'male', # gender of the voice speaking in the reference audio (reference_audio_path): male or female
        output_path = 'my_voice_cloned_and_translated_to_portuguese.wav'
    )

asyncio.run(main())
```