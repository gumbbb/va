#!/usr/bin/env python3
"""
Pipeline from WAV file: load input_test/*.wav -> STT -> Orchestrator (stream) -> TTS.
Use instead of main.py when no microphone; input audio from jetson_test/input_test. No output file.
"""
import os
import sys
import time
import uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import soundfile as sf
except ImportError:
    print("Install: pip install soundfile (or uv add soundfile)")
    sys.exit(1)

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

from stt import STTProcessor
from tts import TTSProcessor
from orchestrator_client import send_and_stream

DEVICE_SAMPLE_RATE = 48000
INPUT_DIR = os.environ.get("INPUT_DIR", "input_test")
TTS_CHUNK_COUNT = int(os.environ.get("TTS_CHUNK_COUNT", "5"))


def _find_input_wav(input_dir):
    if not os.path.isdir(input_dir):
        return None
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".wav"):
            return os.path.join(input_dir, name)
    return None


def run():
    root = os.path.dirname(os.path.abspath(__file__))
    input_dir = INPUT_DIR if os.path.isabs(INPUT_DIR) else os.path.join(root, INPUT_DIR)
    input_wav = os.environ.get("INPUT_WAV")
    if not input_wav and len(sys.argv) > 1:
        input_wav = sys.argv[1]
    if not input_wav:
        input_wav = _find_input_wav(input_dir)
    if not input_wav:
        input_wav = os.path.join(input_dir, "input.wav")
    if not os.path.isfile(input_wav):
        print(f"[main_file] Input WAV not found: {input_wav}")
        print(f"  Put a WAV file in {input_dir} or set INPUT_WAV=/path/to/file.wav")
        sys.exit(1)

    print(f"[main_file] Input: {input_wav}")

    audio, file_sr = sf.read(input_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    stt = STTProcessor(device_sample_rate=file_sr, model_name="small", language=os.getenv("LANG", "en"))
    tts = TTSProcessor(DEVICE_SAMPLE_RATE)
    print("[main_file] STT and TTS loaded")

    print("[main_file] STT start")
    t_stt_start = time.time()
    out = stt.transcribe(audio_int16)
    t_stt_end = time.time()
    text = out.get("text", "").strip()
    lang = out.get("language")
    print(f"[main_file] STT done in {t_stt_end - t_stt_start:.2f}s: '{text}' (lang={lang})")
    if not text:
        print("[main_file] No speech detected, exit")
        sys.exit(0)

    segment_index = [0]
    orchestrator_segments = []

    def on_segment(segment):
        t_receive = time.time()
        seg_num = segment_index[0]
        segment_index[0] += 1
        preview = (segment.strip()[:50] + "…") if len(segment.strip()) > 50 else segment.strip()
        orchestrator_segments.append(segment)
        if seg_num == 0:
            print(f"[main_file] TTS first segment received at {t_receive - t_stt_start:.2f}s from pipeline start")
        print(f"[main_file] Orchestrator segment {seg_num + 1} full: '{segment.strip()}'")
        seg_audio, sr = tts.kokoro.create(
            text=segment.strip(),
            voice="af_heart",
            speed=1.0,
            lang=tts._map_lang(lang),
            is_phonemes=False,
            trim=True,
        )
        t_play_start = time.time()
        if HAS_SOUNDDEVICE:
            audio_int16 = np.clip(seg_audio * 32767, -32768, 32767).astype(np.int16)
            sd.play(audio_int16, sr)
        print(f"[main_file] TTS segment {seg_num + 1}: received → start play in {t_play_start - t_receive:.2f}s: '{preview}'")
        if HAS_SOUNDDEVICE:
            sd.wait()

    orchestrator_url = (os.environ.get("ORCHESTRATOR_URL") or "").strip()
    if orchestrator_url:
        print("[main_file] Orchestrator request start")
        t_orch_start = time.time()
        session_id = f"voice-{uuid.uuid4().hex[:12]}"
        send_and_stream(text, session_id, on_segment, n_chunks=TTS_CHUNK_COUNT)
        t_orch_end = time.time()
        full_response = "".join(orchestrator_segments).strip()
        if full_response:
            print(f"[main_file] Orchestrator full response: '{full_response}'")
        print(f"[main_file] Orchestrator response done in {t_orch_end - t_orch_start:.2f}s")
    else:
        on_segment(text)


if __name__ == "__main__":
    run()
