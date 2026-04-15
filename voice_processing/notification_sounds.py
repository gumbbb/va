import os
import numpy as np
import soundfile as sf

from resampler import Resampler

_ROOT = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(_ROOT, "agent_assets")
KEYS = ("listening", "sending", "timeout", "warning")


def _load_cache():
    cache = {}
    print(f"[notification_sounds] Sounds dir: {SOUNDS_DIR}")
    if not os.path.isdir(SOUNDS_DIR):
        print(f"[notification_sounds] Dir not found, no sounds loaded")
        return cache
    for key in KEYS:
        path = os.path.join(SOUNDS_DIR, key + ".wav")
        if os.path.isfile(path):
            try:
                data, fs = sf.read(path, dtype="float32")
                if data.ndim > 1:
                    data = data[:, 0]
                cache[key] = (data, fs)
                print(f"[notification_sounds] Loaded OK: {key} -> {path} (sr={fs}, samples={len(data)})")
            except Exception as e:
                print(f"[notification_sounds] Load error {path}: {e}")
        else:
            print(f"[notification_sounds] Missing: {path}")
    return cache

_CACHE = _load_cache()


def play_into_queue(key, output_queue, chunk_size, device_sample_rate):
    if key not in _CACHE:
        print(f"[notification_sounds] play_into_queue: key '{key}' not in cache")
        return
    audio, fs = _CACHE[key]
    if fs != device_sample_rate:
        r = Resampler(fs, device_sample_rate)
        audio = r.resample(audio)
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=0)
        elif len(chunk) > chunk_size:
            chunk = chunk[:chunk_size]
        output_queue.put(chunk.reshape(chunk_size, 1).astype(np.float32))


def play(key):
    if key not in _CACHE:
        return
    try:
        import sounddevice as sd
    except ImportError:
        return
    audio, fs = _CACHE[key]
    sd.play(audio, fs, blocking=True)
