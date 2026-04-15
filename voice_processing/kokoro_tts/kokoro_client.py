from kokoro_onnx import Kokoro, SAMPLE_RATE
from kokoro_onnx.tokenizer import Tokenizer
import soundfile as sf
from openai import OpenAI
import asyncio
import queue
import os
import sys
import numpy as np
from pathlib import Path

try:
    from misaki import ja
    MISAKI_AVAILABLE = True
except ImportError:
    MISAKI_AVAILABLE = False

class MinimalBufferStreamTTS:
    def __init__(self, api_key, voice='af_heart', speed=1, min_chars=5, 
                 save_audio=False, output_dir='output_audio',
                 model_path='kokoro-v1.0.onnx',
                 voices_path='voices-v1.0.bin',
                 use_cuda=True, use_quantized=False,
                 audio_buffer_size=4096):
        self.client = OpenAI(api_key=api_key)
        
        if use_quantized and not model_path.startswith('kokoro-v0_19'):
            print("⚠️ Warning: use_quantized=True but model_path doesn't look like quantized model")
            print("   Quantized models: kokoro-v0_19.int8.onnx (88MB) or kokoro-v0_19.fp16.onnx (169MB)")
        
        self.kokoro = Kokoro(model_path, voices_path)
        
        # Log GPU provider status
        session_providers = self.kokoro.sess.get_providers()
        print(f"\n🔍 ONNX Runtime Providers: {session_providers}")
        if "CUDAExecutionProvider" in session_providers:
            print("✅ Using CUDA GPU for TTS inference")
        elif "TensorrtExecutionProvider" in session_providers:
            print("✅ Using TensorRT GPU for TTS inference")
        else:
            print("⚠️ Using CPU for TTS inference (GPU not available)")
        
        self.tokenizer = Tokenizer()
        
        if MISAKI_AVAILABLE:
            try:
                self.misaki_g2p = ja.JAG2P()
                print("✅ Using Misaki G2P for Japanese (better accuracy, no Chinese confusion)")
            except Exception as e:
                print(f"⚠️ Misaki G2P initialization failed: {e}, falling back to phonemizer")
                self.misaki_g2p = None
        else:
            self.misaki_g2p = None
            print("ℹ️ Misaki G2P not available. Install with: pip install 'misaki[ja]'")
            print("   Using phonemizer (may confuse Japanese/Chinese characters)")
        
        self.voice = voice
        self.speed = speed
        self.text_queue = asyncio.Queue()
        self.audio_queue = asyncio.Queue()
        self.char_buffer = ""
        self.min_chars = min_chars
        self.save_audio = save_audio
        self.output_dir = output_dir
        self.audio_counter = 0
        self.sample_rate = SAMPLE_RATE
        self.audio_buffer_size = audio_buffer_size
        
        self.audio_player = self._init_audio_player()
        if self.audio_player:
            print(f"🔊 Using audio player: {self.audio_player}")
        else:
            print("⚠️ No audio player available")
        
        if self.save_audio:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def _init_audio_player(self):
        try:
            import sounddevice as sd
            return 'sounddevice'
        except ImportError:
            pass
        
        try:
            import pygame
            # Increase buffer size significantly to reduce ALSA underrun
            # Larger buffer = more tolerance for timing variations
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=4096)
            return 'pygame'
        except ImportError:
            pass
        
        return None
    
    def _play_audio_sounddevice(self, audio_data):
        import sounddevice as sd
        print(f"🔊 [sounddevice] Playing audio chunk ({len(audio_data)} samples)")
        try:
            sd.play(audio_data.astype(np.float32), samplerate=self.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"⚠️ Sounddevice playback error: {e}")
    
    def _play_audio_pygame(self, audio_data):
        import pygame
        import io
        import wave
        
        print(f"🔊 [pygame] Playing audio chunk ({len(audio_data)} samples)")
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        sound = pygame.mixer.Sound(wav_buffer)
        channel = sound.play()
        return channel
    
    def _cleanup_audio_stream(self):
        pass
        
    def stream_openai_response(self, user_question, system_prompt=None):
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant. Respond in Japanese."},
            {"role": "user", "content": user_question}
        ]
        
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            temperature=0.7
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def process_with_kokoro_stream(self, text):
        if not text or len(text) == 0:
            return
            
        text_clean = text.strip()
        print(f"📝 Processing text ({len(text_clean)} chars): {text_clean[:100]}..." if len(text_clean) > 100 else f"📝 Processing text: {text_clean}")
        print(f"   Voice: {self.voice}, Speed: {self.speed}")
        
        try:
            if self.misaki_g2p is not None:
                print("   Using Misaki G2P for Japanese text → phonemes")
                phonemes, _ = self.misaki_g2p(text_clean)
                print(f"🔤 Generated phonemes ({len(phonemes)} chars): {phonemes[:200]}..." if len(phonemes) > 200 else f"🔤 Generated phonemes: {phonemes}")
                
                stream = self.kokoro.create_stream(
                    phonemes,
                    voice=self.voice,
                    speed=self.speed,
                    is_phonemes=True
                )
            else:
                print("   Using phonemizer (lang=ja)")
                phonemes = self.tokenizer.phonemize(text_clean, lang="ja", norm=True)
                print(f"🔤 Generated phonemes ({len(phonemes)} chars): {phonemes[:200]}..." if len(phonemes) > 200 else f"🔤 Generated phonemes: {phonemes}")
                
                stream = self.kokoro.create_stream(
                    text_clean, 
                    voice=self.voice, 
                    speed=self.speed, 
                    lang="ja"
                )
            
            chunk_count = 0
            async for samples, sample_rate in stream:
                chunk_count += 1
                print(f"🔊 Generated audio chunk {chunk_count} ({len(samples)} samples, {sample_rate}Hz)", flush=True)
                
                # Put audio into queue for async playback
                await self.audio_queue.put((samples, sample_rate))
                
                if self.save_audio:
                    output_file = os.path.join(self.output_dir, f'audio_{self.audio_counter:04d}.wav')
                    sf.write(output_file, samples, sample_rate)
                    print(f"   → Saved: {output_file}")
                    self.audio_counter += 1
                
        except Exception as e:
            import traceback
            print(f"⚠️ TTS Error: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
    
    async def audio_playback_worker(self):
        if self.audio_player == 'sounddevice':
            while True:
                try:
                    item = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                    if item is None:
                        break
                    samples, sample_rate = item
                    await asyncio.to_thread(self._play_audio_sounddevice, samples)
                    self.audio_queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"⚠️ Audio playback error: {e}")
                    self.audio_queue.task_done()
        elif self.audio_player == 'pygame':
            import pygame
            current_channel = None
            while True:
                try:
                    # Wait for current channel to finish before playing next
                    if current_channel is not None and current_channel.get_busy():
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Get new audio chunk
                    try:
                        item = await asyncio.wait_for(self.audio_queue.get(), timeout=0.01)
                        if item is None:
                            # Wait for current channel to finish
                            if current_channel is not None:
                                while current_channel.get_busy():
                                    await asyncio.sleep(0.05)
                            break
                        samples, sample_rate = item
                        current_channel = await asyncio.to_thread(self._play_audio_pygame, samples)
                        self.audio_queue.task_done()
                    except asyncio.TimeoutError:
                        continue
                except Exception as e:
                    print(f"⚠️ Audio playback error: {e}")
                    if not self.audio_queue.empty():
                        self.audio_queue.task_done()
        else:
            # No audio player, just drain the queue
            while True:
                try:
                    item = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                    if item is None:
                        break
                    self.audio_queue.task_done()
                except asyncio.TimeoutError:
                    continue
    
    async def tts_worker_async(self):
        while True:
            try:
                text = await asyncio.wait_for(self.text_queue.get(), timeout=0.1)
                if text is None:
                    break
                if text:
                    await self.process_with_kokoro_stream(text)
                    self.text_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"❌ Worker Error: {e}")
    
    async def process_realtime_async(self, user_question, system_prompt=None):
        print(f"👤 User: {user_question}\n")
        print("🤖 AI Response (streaming):\n")
        
        self.char_buffer = ""
        self.audio_counter = 0
        
        tts_task = asyncio.create_task(self.tts_worker_async())
        audio_task = asyncio.create_task(self.audio_playback_worker())
        
        try:
            for chunk in self.stream_openai_response(user_question, system_prompt):
                print(chunk, end='', flush=True)
                
                for char in chunk:
                    self.char_buffer += char
                    
                    if len(self.char_buffer) >= self.min_chars or char in ['。', '！', '？', '、', '，', '\n', '.', '!', '?', ',']:
                        if self.char_buffer.strip():
                            await self.text_queue.put(self.char_buffer.strip())
                        self.char_buffer = ""
            
            if self.char_buffer.strip():
                await self.text_queue.put(self.char_buffer.strip())
            
            await self.text_queue.put(None)
            await tts_task
            
            await self.audio_queue.put(None)
            await audio_task
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            await self.text_queue.join()
            await self.audio_queue.join()
        
        if self.save_audio:
            print(f"\n✅ Done! Audio files saved in '{self.output_dir}' directory")
        else:
            print("\n✅ Done!")
    
    def process_realtime(self, user_question, system_prompt=None):
        asyncio.run(self.process_realtime_async(user_question, system_prompt))

def load_env_file(env_path=None):
    """
    Load environment variables from .env file
    """
    if env_path is None:
        env_path = Path(__file__).parent / '.env'
    else:
        env_path = Path(env_path)
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

def get_api_key():
    """
    Get OpenAI API key from:
    1. Command line argument (--api-key)
    2. Environment variable OPENAI_API_KEY
    3. .env file
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        load_env_file()
        api_key = os.getenv('OPENAI_API_KEY')
    return api_key

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Realtime TTS with OpenAI and Kokoro ONNX')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (or use OPENAI_API_KEY env var or .env file)')
    parser.add_argument('--env-file', type=str, help='Path to .env file (default: .env in script directory)')
    parser.add_argument('--question', type=str, help='User question (or will prompt)')
    parser.add_argument('--voice', type=str, default='af_heart', help='Kokoro voice')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed')
    parser.add_argument('--min-chars', type=int, default=5, help='Minimum characters before TTS (default: 5 to reduce TTS calls)')
    parser.add_argument('--audio-buffer-size', type=int, default=4096, help='Audio buffer size for sounddevice (default: 4096)')
    parser.add_argument('--save-audio', action='store_true', help='Save audio files to disk')
    parser.add_argument('--output-dir', type=str, default='output_audio', help='Output directory for audio files')
    parser.add_argument('--model-path', type=str, default='kokoro-v1.0.onnx', 
                        help='ONNX model path (kokoro-v1.0.onnx or kokoro-v1.1.onnx for newer version)')
    parser.add_argument('--voices-path', type=str, default='voices-v1.0.bin', 
                        help='Voices file path (voices-v1.0.bin or voices-v1.1.bin for newer version)')
    parser.add_argument('--use-quantized', action='store_true', help='Use quantized model (INT8/FP16)')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use CUDA if available')
    
    args = parser.parse_args()
    
    if args.env_file:
        load_env_file(args.env_file)
    
    api_key = args.api_key or get_api_key()
    
    if not api_key:
        print("❌ Error: OpenAI API key not found!")
        print("   Please provide it via:")
        print("   1. --api-key argument")
        print("   2. OPENAI_API_KEY environment variable")
        print("   3. OPENAI_API_KEY in .env file")
        sys.exit(1)
    
    tts = MinimalBufferStreamTTS(
        api_key=api_key,
        voice=args.voice,
        speed=args.speed,
        min_chars=args.min_chars,
        save_audio=args.save_audio,
        output_dir=args.output_dir,
        model_path=args.model_path,
        voices_path=args.voices_path,
        use_cuda=args.use_cuda,
        use_quantized=args.use_quantized,
        audio_buffer_size=args.audio_buffer_size
    )
    
    if args.question:
        user_question = args.question
    else:
        user_question = input("Enter your question: ")
    
    tts.process_realtime(user_question)

if __name__ == "__main__":
    main()
