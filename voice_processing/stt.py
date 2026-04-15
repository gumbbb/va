import io
import os
import re
import numpy as np
import soundfile as sf
from pydub import AudioSegment

if os.environ.get("WHISPER_OFFLINE") or os.environ.get("WHISPER_BASE_MODEL_PATH"):
    os.environ["HF_HUB_OFFLINE"] = "1"

from whisper_trt import load_trt_model, set_cache_dir

STT_SAMPLE_RATE = 16000
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "whisper_trt")


def _pad_audio(audio_data, sample_width=2, channels=1):
    remainder = len(audio_data) % (sample_width * channels)
    if remainder != 0:
        padding = b"\x00" * (sample_width * channels - remainder)
        return audio_data + padding
    return audio_data


def resample_audio(audio_int16, original_sr, target_sr):
    if not isinstance(audio_int16, np.ndarray):
        audio_int16 = np.frombuffer(audio_int16, dtype=np.int16)
    padded = _pad_audio(audio_int16.tobytes(), 2, 1)
    try:
        seg = AudioSegment(data=padded, sample_width=2, frame_rate=original_sr, channels=1)
        seg = seg.set_frame_rate(target_sr)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    except Exception:
        n = int(len(audio_int16) * target_sr / original_sr)
        samples = np.interp(
            np.linspace(0, len(audio_int16) - 1, n),
            np.arange(len(audio_int16)),
            audio_int16.astype(np.float32) / 32768.0,
        )
    return samples


def _apply_local_whisper_patch(model_name):
    base_path = os.environ.get("WHISPER_BASE_MODEL_PATH")
    if not base_path or not os.path.isfile(base_path):
        return
    import whisper
    _orig_load = whisper.load_model

    def _local_load(name, *args, **kwargs):
        path = base_path if name == model_name else name
        return _orig_load(path, *args, **kwargs)

    whisper.load_model = _local_load


class STTProcessor:
    def __init__(self, device_sample_rate, model_name="small", language=None):
        self.device_sample_rate = device_sample_rate
        self.model_name = model_name
        self.language = language
        trt_path = os.environ.get("WHISPER_TRT_MODEL_PATH")
        if trt_path:
            if not os.path.isfile(trt_path):
                raise FileNotFoundError(f"WHISPER_TRT_MODEL_PATH not found: {trt_path}")
            print(f"[STT] Using local TRT model: {trt_path}")
        elif os.path.isdir(_CACHE_DIR):
            set_cache_dir(_CACHE_DIR)
            print(f"[STT] Whisper TRT cache: {_CACHE_DIR}")
        _apply_local_whisper_patch(model_name)
        print(f"[STT] Loading Whisper TRT: {model_name} (lang={language or 'auto'})")
        self.model = load_trt_model(
            model_name,
            path=trt_path if trt_path else None,
            build=False,
        )
        self.model.transcribe(np.zeros(1536, dtype=np.float32), language=language)
        print("[STT] Whisper TRT ready")

    def transcribe(self, audio_int16):
        samples = resample_audio(audio_int16, self.device_sample_rate, STT_SAMPLE_RATE)
        result = self.model.transcribe(samples, language=self.language)
        text = result.get("text", "")
        
        # 1. Try to get from result dict (most common for modern whisper wrappers)
        detected_lang = result.get("language")
        
        # 2. Try to regex from text tags like <|en|>
        if not detected_lang:
            m = re.search(r"<\|([a-z]{2})\|>", text)
            if m:
                detected_lang = m.group(1)
        
        # 3. Fallback to initialized language
        if not detected_lang:
            detected_lang = self.language
            
        clean = re.sub(r"<\|[^|]+\|>", "", text).strip()
        return {"text": clean, "language": detected_lang}
