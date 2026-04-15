import os
import numpy as np
import openwakeword

_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(_ROOT, "agent_assets", "models")
MODEL_CHUNK = 1280
SAMPLE_RATE = 16000

DORA_MODELS = [
    os.path.join(ASSETS, "hey_dora.onnx"),
    os.path.join(ASSETS, "hey_doh_ra.onnx"),
    os.path.join(ASSETS, "hey_dola.onnx"),
]


def _model_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


class WakeWordProcessor:
    def __init__(self, device_sample_rate, detect_prob=0.4):
        print(f"[wake_word] Assets dir: {ASSETS}")
        if not os.path.isdir(ASSETS):
            raise FileNotFoundError(f"Wake word assets directory not found: {ASSETS}")
        existing = [p for p in DORA_MODELS if os.path.isfile(p)]
        for p in DORA_MODELS:
            name = _model_basename(p)
            if p in existing:
                print(f"[wake_word] ONNX loaded: {name} -> {p}")
            else:
                print(f"[wake_word] Missing: {name} -> {p}")
        if not existing:
            raise FileNotFoundError(f"Wake word models not found under {ASSETS}")
        self.device_sample_rate = device_sample_rate
        self.detect_prob = detect_prob
        self._detected = False
        self._last_model_name = None
        self._last_prob = 0.0
        self.buffer = np.array([], dtype=np.int16)
        self.resampler = None
        self._frame_length = MODEL_CHUNK
        self.model = openwakeword.Model(wakeword_models=existing, inference_framework="onnx")
        print(f"[wake_word] All ONNX models ready. Threshold={detect_prob}")

    def update_state(self, audio_bytes):
        chunk = np.frombuffer(audio_bytes, dtype=np.int16)
        if self.resampler is None:
            from resampler import Resampler
            self.resampler = Resampler(self.device_sample_rate, SAMPLE_RATE)
        chunk_f32 = chunk.astype(np.float32) / 32768.0
        resampled = self.resampler.resample(chunk_f32)
        resampled_int16 = (resampled * 32767).astype(np.int16)
        self.buffer = np.concatenate((self.buffer, resampled_int16))
        if self._detected:
            return
        if len(self.buffer) < self._frame_length:
            return
        to_process = self.buffer[:self._frame_length].copy()
        self.buffer = self.buffer[self._frame_length:]
        pred = self.model.predict(to_process)
        if isinstance(pred, dict):
            if pred:
                best = max(pred.items(), key=lambda x: x[1])
                model_name = _model_basename(best[0]) if "/" in best[0] or "\\" in best[0] else best[0]
                prob = best[1]
                self._last_model_name = model_name
                self._last_prob = prob
                if prob >= self.detect_prob:
                    self._detected = True
                    print(f"[wake_word] Detected: {model_name} prob={prob:.3f} (threshold={self.detect_prob})")
            else:
                prob = 0.0
        else:
            prob = float(pred) if pred is not None else 0.0
            if prob >= self.detect_prob:
                self._detected = True
                print(f"[wake_word] Detected: prob={prob:.3f} (threshold={self.detect_prob})")

    def get_detected(self):
        out = self._detected
        self._detected = False
        self._last_model_name = None
        self._last_prob = 0.0
        return out
