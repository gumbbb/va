#!/usr/bin/env python3
"""
Test Whisper TRT with audio file and verify GPU usage.
Supports full-audio mode and chunk-based streaming simulation.

Usage:
  python test_whisper_gpu.py <audio_file> [model_name] [language]
  python test_whisper_gpu.py /path/to/audio.wav base ja --chunk --chunk-sec 2.0
"""
import argparse
import os
import re
import sys
import time

import numpy as np
from pydub import AudioSegment

sys.path.insert(0, "/app")

WHISPER_SAMPLE_RATE = 16000


def load_audio_16k(audio_path):
    audio = AudioSegment.from_file(audio_path)
    if audio.channels > 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != WHISPER_SAMPLE_RATE:
        audio = audio.set_frame_rate(WHISPER_SAMPLE_RATE)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return samples, len(samples) / WHISPER_SAMPLE_RATE


def test_whisper_full(audio_path, model_name, language, model):
    print("\n--- Mode: FULL (batch) ---")
    samples, duration = load_audio_16k(audio_path)
    print(f"Audio: {duration:.2f}s, {WHISPER_SAMPLE_RATE}Hz")
    print("Transcribing full audio...")
    t0 = time.time()
    result = model.transcribe(samples, language=language)
    infer_time = time.time() - t0
    text = result.get("text", "")
    clean = re.sub(r"<\|[^|]+\|>", "", text).strip()
    print(f"Result: {clean}")
    print(f"Inference time: {infer_time:.2f}s (RTF: {infer_time/duration:.3f})")
    return clean, infer_time


def test_whisper_chunked(audio_path, model_name, language, model, chunk_sec=1.0):
    print("\n--- Mode: CHUNK (streaming simulation) ---")
    samples, duration = load_audio_16k(audio_path)
    print(f"Audio: {duration:.2f}s, chunk_size: {chunk_sec}s")
    chunk_samples = int(chunk_sec * WHISPER_SAMPLE_RATE)
    num_chunks = (len(samples) + chunk_samples - 1) // chunk_samples
    partial_texts = []
    total_infer = 0.0
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, len(samples))
        chunk = samples[start:end]
        t0 = time.time()
        result = model.transcribe(chunk, language=language)
        dt = time.time() - t0
        total_infer += dt
        text = result.get("text", "")
        clean = re.sub(r"<\|[^|]+\|>", "", text).strip()
        partial_texts.append(clean)
        t_start = start / WHISPER_SAMPLE_RATE
        t_end = end / WHISPER_SAMPLE_RATE
        print(f"  Chunk {i+1}/{num_chunks} [{t_start:.1f}s-{t_end:.1f}s] ({dt:.3f}s): '{clean}'")
    full_text = " ".join(t for t in partial_texts if t)
    print(f"\nConcatenated: {full_text}")
    print(f"Total inference: {total_infer:.2f}s (RTF: {total_infer/duration:.3f})")
    return full_text, total_infer


def test_whisper_gpu(
    audio_path, model_name="base", language="ja", chunk_mode=False, chunk_sec=1.0
):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return

    print("=" * 60)
    print(f"Loading Whisper TRT model: {model_name} (language={language})")
    print("=" * 60)

    from whisper_trt import load_trt_model

    t0 = time.time()
    model = load_trt_model(model_name, build=True, verbose=True)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    if chunk_mode:
        test_whisper_chunked(audio_path, model_name, language, model, chunk_sec)
    else:
        test_whisper_full(audio_path, model_name, language, model)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Whisper TRT - full or chunk-based streaming"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="/home/acevn/ai-demo-edge-main/test_japanese.wav",
        help="Path to audio file",
    )
    parser.add_argument("model_name", nargs="?", default="base", help="Model name")
    parser.add_argument("language", nargs="?", default="ja", help="Language code")
    parser.add_argument(
        "-c", "--chunk", action="store_true", help="Chunk-based streaming mode"
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=1.0,
        help="Chunk size in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    test_whisper_gpu(
        args.audio_file,
        args.model_name,
        args.language,
        chunk_mode=args.chunk,
        chunk_sec=args.chunk_sec,
    )
