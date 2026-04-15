import os
import time
import numpy as np
from kokoro_onnx import Kokoro

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

KOKORO_SAMPLE_RATE = 24000
LANGUAGE_MAP = {
    "en": "en-us",
    "ja": "ja",
    "vi": "vi",
    "zh": "zh",
    "fr": "fr",
    "es": "es",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "ko": "ko",
    "hi": "hi",
}


def _model_paths():
    onnx_env = os.environ.get("KOKORO_ONNX_PATH")
    voices_env = os.environ.get("KOKORO_VOICES_PATH")
    if onnx_env and voices_env and os.path.isfile(onnx_env) and os.path.isfile(voices_env):
        return onnx_env, voices_env
    _root = os.path.dirname(os.path.abspath(__file__))
    for base in [_root, os.path.join(_root, ".."), "/app", os.getcwd()]:
        for prefix in ["kokoro_tts", "kokoro-onnx"]:
            d = os.path.join(base, prefix) if base else prefix
            onnx = os.path.join(d, "kokoro-v1.0.onnx")
            voices = os.path.join(d, "voices-v1.0.bin")
            if os.path.isfile(onnx) and os.path.isfile(voices):
                return onnx, voices
    return "kokoro-v1.0.onnx", "voices-v1.0.bin"


class TTSProcessor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        onnx_path, voices_path = _model_paths()
        print(f"[TTS] Kokoro onnx path: {onnx_path}")
        print(f"[TTS] Kokoro voices path: {voices_path}")
        if not os.path.isfile(onnx_path) or not os.path.isfile(voices_path):
            raise FileNotFoundError(f"Kokoro model not found: onnx={onnx_path}, voices={voices_path}")
        self.kokoro = Kokoro(onnx_path, voices_path)
        print(f"[TTS] Kokoro loaded OK: onnx={onnx_path}, voices={voices_path}")

    def _map_lang(self, language):
        if language is None:
            return "en-us"
        return LANGUAGE_MAP.get(language, "en-us")

    def speech(self, text, language=None, output_queue=None, chunk_size=None, device_sample_rate=None):
        if not text or not text.strip():
            return
        lang = self._map_lang(language)
        print(f"[TTS] Speaking: repr={text!r} len={len(text)} (lang={lang})")
        t0 = time.time()
        try:
            audio, sr = self.kokoro.create(
                text=text.strip(),
                voice="af_heart",
                speed=1.0,
                lang=lang,
                is_phonemes=False,
                trim=True,
            )
            if output_queue is not None and chunk_size is not None and device_sample_rate is not None:
                from resampler import Resampler
                if sr != device_sample_rate:
                    r = Resampler(sr, device_sample_rate)
                    audio = r.resample(audio)
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=0)
                    elif len(chunk) > chunk_size:
                        chunk = chunk[:chunk_size]
                    output_queue.put(chunk.reshape(chunk_size, 1).astype(np.float32))
            else:
                audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                if HAS_SOUNDDEVICE:
                    sd.play(audio_int16, sr)
                    sd.wait()
                else:
                    print("[TTS] sounddevice not available, skip playback")
        except Exception as e:
            print(f"[TTS] Error: {e}")
            import traceback
            traceback.print_exc()
        print(f"[TTS] Done ({time.time() - t0:.2f}s)")
