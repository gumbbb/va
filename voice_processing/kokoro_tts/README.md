# Tóm tắt các file examples trong kokoro-onnx

## 1. File cơ bản

- `play.py` — Phát audio trực tiếp qua sounddevice
- `save.py` — Tạo audio và lưu ra file WAV

## 2. File theo ngôn ngữ

- `japanese.py` — Tiếng Nhật, dùng Misaki G2P với espeak-ng fallback
- `english.py` — Tiếng Anh, dùng Misaki G2P
- `chinese.py` — Tiếng Trung, dùng Misaki G2P (cần model v1.1-zh)
- `french.py`, `hindi.py`, `italian.py`, `portuguese.py`, `spanish.py` — Các ngôn ngữ khác, dùng EspeakG2P

## 3. File nâng cao — tính năng

- `app.py` — Gradio web app với UI để tạo audio, hỗ trợ blend voice
- `with_stream.py` — Streaming audio (phát từng chunk khi tạo)
- `with_stream_save.py` — Streaming và lưu file
- `with_phonemes.py` — Dùng phonemes trực tiếp thay vì text
- `with_voice.py` — Tạo audio với tất cả giọng có sẵn
- `with_blending.py` — Trộn 2 giọng (50/50)
- `podcast.py` — Tạo podcast với nhiều giọng, thêm khoảng lặng ngẫu nhiên

## 4. File nâng cao — tối ưu hóa

- `with_cuda.py` — Sử dụng CUDA/GPU (cần cài CUDA và CUDNN)
- `with_quant.py` — Dùng model quantized (INT8 88MB hoặc FP16 169MB)
- `with_provider.py` — Chỉ định ONNX execution provider (CPU/GPU)
- `with_session.py` — Tùy chỉnh ONNX session (số thread, providers)

## 5. File nâng cao — cấu hình

- `with_log.py` — Bật debug logging
- `with_espeak_data.py` — Cấu hình đường dẫn espeak-ng data
- `with_espeak_lib.py` — Cấu hình đường dẫn espeak-ng library

## Điểm chung

- Tất cả đều cần download model files: `kokoro-v1.0.onnx` và `voices-v1.0.bin`
- Hầu hết dùng `soundfile` để lưu hoặc `sounddevice` để phát
- Các file ngôn ngữ dùng Misaki hoặc EspeakG2P để chuyển text → phonemes
- API chính: `Kokoro.create()` và `Kokoro.create_stream()` (async)

Các ví dụ này minh họa cách sử dụng Kokoro TTS từ cơ bản đến nâng cao, hỗ trợ nhiều ngôn ngữ và tùy chọn tối ưu hóa.

## kokoro_client.py

### Hướng dẫn cài đặt

#### 1. Cài đặt Python dependencies

**Với uv (khuyến nghị):**
```bash
cd /path/to/project/tts
uv sync
```

**Hoặc với pip:**
```bash
pip install kokoro-onnx openai soundfile sounddevice pygame numpy
pip install 'misaki[ja]' 'fugashi[unidic-lite]' unidic-lite
```

#### 2. Cài đặt MeCab (cho Misaki G2P - tiếng Nhật)

**Trên Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
```

**Trên macOS:**
```bash
brew install mecab
```

#### 3. Download UniDic dictionary

Sau khi cài đặt MeCab và fugashi, cần download UniDic dictionary:

```bash
python -m unidic download
```

Hoặc trong Python:
```python
import unidic
unidic.download()
```

**Lưu ý:** Dictionary này khá lớn (~526MB), quá trình download có thể mất vài phút.

#### 4. Download Kokoro model files

Download model và voices files từ GitHub releases:

```bash
# Model v1.0 (khuyến nghị)
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Hoặc model v1.1 (mới hơn)
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1.bin
```

Đặt các file này trong cùng thư mục với `kokoro_client.py` hoặc chỉ định đường dẫn bằng `--model-path` và `--voices-path`.

#### 5. Cấu hình OpenAI API key

Có 3 cách để cung cấp API key:

**Cách 1: Command line argument**
```bash
python kokoro_client.py --api-key sk-your-api-key-here
```

**Cách 2: Environment variable**
```bash
export OPENAI_API_KEY=sk-your-api-key-here
python kokoro_client.py
```

**Cách 3: .env file**
Tạo file `.env` trong cùng thư mục với `kokoro_client.py`:
```
OPENAI_API_KEY=sk-your-api-key-here
```

#### 6. Kiểm tra cài đặt

Test xem MeCab có hoạt động không:
```bash
python -c "import fugashi; tagger = fugashi.Tagger(); print('✅ MeCab OK')"
```

Nếu gặp lỗi, đảm bảo đã:
- Cài đặt MeCab system package
- Download UniDic dictionary (`python -m unidic download`)
- Cài đặt `fugashi[unidic-lite]` và `unidic-lite`

#### 7. Troubleshooting

**Lỗi: "Failed initializing MeCab"**
- Đảm bảo đã cài MeCab: `sudo apt-get install mecab libmecab-dev`
- Download UniDic: `python -m unidic download`
- Reinstall fugashi: `pip install --force-reinstall 'fugashi[unidic-lite]'`

**Lỗi: "Misaki G2P initialization failed"**
- Cài đặt MeCab system package (xem bước 2)
- Download UniDic dictionary (xem bước 3)
- Code sẽ tự động fallback về phonemizer nếu Misaki không hoạt động

**Audio không phát được**
- Cài đặt `sounddevice`: `pip install sounddevice`
- Hoặc cài đặt `pygame`: `pip install pygame`
- Trên Linux có thể cần: `sudo apt-get install portaudio19-dev` (cho sounddevice)

File này chứa class `MinimalBufferStreamTTS` - một client tích hợp OpenAI API với Kokoro TTS để tạo hệ thống TTS real-time.

### Tính năng chính:
- **Streaming OpenAI Response**: Nhận response từ OpenAI GPT-4o-mini theo dòng (streaming)
- **Real-time TTS**: Chuyển text thành speech ngay khi nhận được, với buffer tối thiểu
- **Minimal Buffer**: Gửi text đến TTS ngay khi đủ số ký tự tối thiểu hoặc gặp dấu câu
- **Multi-threading**: Sử dụng thread riêng để xử lý TTS, không block streaming
- **Audio Playback**: Hỗ trợ phát audio qua sounddevice (ưu tiên) hoặc pygame (fallback)
- **Save Audio**: Tùy chọn lưu các file audio ra disk
- **Japanese Language**: Hỗ trợ tiếng Nhật với Misaki G2P (tự động fallback về phonemizer nếu không có)

### Cách sử dụng:

**Command line:**
```bash
python kokoro_client.py --api-key YOUR_OPENAI_API_KEY --question "Your question here"
```

**Các tham số:**
- `--api-key`: OpenAI API key (hoặc dùng OPENAI_API_KEY env var hoặc .env file)
- `--env-file`: Đường dẫn đến file .env (mặc định: .env trong thư mục script)
- `--question`: Câu hỏi của user (nếu không có sẽ prompt)
- `--voice`: Giọng Kokoro (mặc định: 'af_heart')
- `--speed`: Tốc độ nói (mặc định: 1.0)
- `--min-chars`: Số ký tự tối thiểu trước khi gửi đến TTS (mặc định: 5)
- `--audio-buffer-size`: Kích thước buffer cho sounddevice (mặc định: 4096)
- `--save-audio`: Lưu các file audio ra disk
- `--output-dir`: Thư mục lưu audio files (mặc định: 'output_audio')
- `--model-path`: Đường dẫn đến file model ONNX (mặc định: 'kokoro-v1.0.onnx')
- `--voices-path`: Đường dẫn đến file voices (mặc định: 'voices-v1.0.bin')
- `--use-quantized`: Sử dụng model quantized (INT8/FP16)
- `--use-cuda`: Sử dụng CUDA nếu có (mặc định: True)

**Ví dụ:**
```bash
# Chạy với câu hỏi trực tiếp (tiếng Nhật)
python kokoro_client.py --api-key sk-xxx --question "こんにちは"

# Chạy với lưu audio
python kokoro_client.py --api-key sk-xxx --question "Hello" --save-audio --output-dir my_audio

# Tùy chỉnh giọng và tốc độ
python kokoro_client.py --api-key sk-xxx --voice af_sarah --speed 1.2

# Tăng min-chars để có câu dài hơn (ít TTS calls hơn)
python kokoro_client.py --api-key sk-xxx --min-chars 15

# Dùng model v1.1 mới hơn
python kokoro_client.py --api-key sk-xxx --model-path kokoro-v1.1.onnx --voices-path voices-v1.1.bin

# Dùng .env file thay vì --api-key
python kokoro_client.py --env-file .env --question "こんにちは"
```

### Cấu trúc class:

**MinimalBufferStreamTTS:**
- `__init__()`: Khởi tạo với API key, voice, speed, và các tùy chọn
- `stream_openai_response()`: Stream response từ OpenAI API
- `process_with_kokoro()`: Xử lý text với Kokoro TTS pipeline
- `process_realtime()`: Hàm chính để xử lý real-time streaming và TTS
- `_play_audio()`: Phát audio từ numpy array
- `_init_audio_player()`: Khởi tạo audio player (sounddevice/pygame)

### Dependencies:
- `kokoro-onnx`: Kokoro TTS ONNX runtime
- `openai`: OpenAI client
- `soundfile`: Lưu audio files
- `sounddevice` hoặc `pygame`: Phát audio
- `numpy`: Xử lý audio data
- `misaki[ja]`: Misaki G2P cho tiếng Nhật (khuyến nghị)
- `fugashi[unidic-lite]`: MeCab wrapper cho Python
- `unidic-lite`: UniDic dictionary cho MeCab
- `asyncio`: Async/await support