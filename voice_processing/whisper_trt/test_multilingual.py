#!/usr/bin/env python3
"""
Test script for Whisper TRT với multilingual support
"""

import numpy as np
from whisper_trt import load_trt_model
import sys
import os
from pydub import AudioSegment

def test_model(model_name, audio_file_path, language=None):
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"Audio file: {audio_file_path}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"{'='*60}")
    
    try:
        print(f"\nLoading model {model_name}...")
        model = load_trt_model(model_name, build=True, verbose=True)
        print("Model loaded successfully!")
        
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found: {audio_file_path}")
            return None
        
        print(f"\nLoading audio file...")
        audio = AudioSegment.from_file(audio_file_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        
        print(f"Transcribing...")
        result = model.transcribe(audio_data, language=language)
        
        transcript = result['text']
        
        # Extract language from transcript (format: <|ja|><|transcribe|><|notimestamps|>text...)
        import re
        detected_language = None
        lang_match = re.search(r'<\|([a-z]{2})\|>', transcript)
        if lang_match:
            detected_language = lang_match.group(1)
        elif language:
            detected_language = language
        
        # Remove special tokens from transcript for display
        clean_transcript = re.sub(r'<\|[^|]+\|>', '', transcript).strip()
        
        print(f"\n✅ Transcription: {clean_transcript}")
        if detected_language:
            print(f"🌐 Detected language: {detected_language}")
        
        return clean_transcript
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_multilingual.py <audio_file> [model_name] [language]")
        print("\nExamples:")
        print("  python test_multilingual.py test.wav base")
        print("  python test_multilingual.py test.wav base ja")
        print("  python test_multilingual.py test.wav base vi")
        print("  python test_multilingual.py test.wav base en")
        print("\nAvailable models:")
        print("  Multilingual: tiny, base, small")
        print("  English-only: tiny.en, base.en, small.en")
        print("\nAvailable languages:")
        print("  auto, en, ja, vi, zh, ... (100+ languages)")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"
    language = sys.argv[3] if len(sys.argv) > 3 else None
    
    test_model(model_name, audio_file, language)

