import os
import numpy as np
import torch
from silero_vad.utils_vad import OnnxWrapper

from resampler import Resampler

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_ROOT, "agent_assets", "models", "silero_vad.onnx")
MODEL_CHUNK = 512
VAD_SAMPLE_RATE = 16000


class VADProcessor:
    def __init__(self, device_sample_rate):
        print(f"[VAD] Model path: {MODEL_PATH}")
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"VAD model not found: {MODEL_PATH}")
        self.device_sample_rate = device_sample_rate
        self.model = OnnxWrapper(MODEL_PATH, True)
        print(f"[VAD] ONNX model loaded OK: {MODEL_PATH}")
        self.buffer = np.array([], dtype=np.float32)
        self.resampler = None

    def update_buffer(self, audio_bytes):
        if self.resampler is None:
            self.resampler = Resampler(self.device_sample_rate, VAD_SAMPLE_RATE)
        audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        resampled = self.resampler.resample(audio_f32)
        self.buffer = np.concatenate((self.buffer, resampled))

    def get_prob(self):
        if len(self.buffer) < MODEL_CHUNK:
            return 0.0
        chunk = self.buffer[:MODEL_CHUNK].copy()
        self.buffer = self.buffer[MODEL_CHUNK:]
        tensor = torch.from_numpy(chunk).float()
        out = self.model(tensor, VAD_SAMPLE_RATE)
        if out.numel() == 1:
            return out.item()
        return out[0].item()

    def empty_buffer(self):
        self.buffer = np.array([], dtype=np.float32)
