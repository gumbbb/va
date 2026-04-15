#!/usr/bin/env python3
"""
Run TTS only and write audio to a WAV file. No Docker or full pipeline.
Usage: python tts_to_wav.py "Your text here" [output.wav] [--lang en-us]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import soundfile as sf
except ImportError:
    print("Install: pip install soundfile (or uv add soundfile)")
    sys.exit(1)

from tts import TTSProcessor, KOKORO_SAMPLE_RATE


def main():
    if len(sys.argv) < 2:
        print('Usage: python tts_to_wav.py "Your text" [output.wav] [--lang en-us]')
        sys.exit(1)
    text = sys.argv[1].strip()
    if not text:
        print("Empty text")
        sys.exit(1)
    args = sys.argv[2:]
    out_path = "output.wav"
    lang = "en-us"
    i = 0
    while i < len(args):
        if args[i] == "--lang" and i + 1 < len(args):
            lang = args[i + 1]
            i += 2
            continue
        if not args[i].startswith("-"):
            out_path = args[i]
        i += 1
    if not out_path.lower().endswith(".wav"):
        out_path += ".wav"
    tts = TTSProcessor(KOKORO_SAMPLE_RATE)
    audio, sr = tts.kokoro.create(
        text=text,
        voice="af_heart",
        speed=1.0,
        lang=lang,
        is_phonemes=False,
        trim=True,
    )
    sf.write(out_path, audio, sr)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
