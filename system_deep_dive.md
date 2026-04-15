# Complete System Deep Dive: Voice Pipeline & WebSocket Architecture

> **Edge AI Orchestration System** — NVIDIA Jetson  
> Services: `voice_processing/` (embedded voice client) + `orchestrator-on-edge/` (FastAPI brain + agents + frontend)

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Service Map & Process Boundaries](#2-service-map--process-boundaries)
3. [Stage 1 — Audio Capture (Full-Duplex Stream)](#3-stage-1--audio-capture-full-duplex-stream)
4. [Stage 2 — Wake Word Detection](#4-stage-2--wake-word-detection)
5. [Stage 3 — Voice Activity Detection (VAD)](#5-stage-3--voice-activity-detection-vad)
6. [Stage 4 — STT Timing State Machine](#6-stage-4--stt-timing-state-machine)
7. [Stage 5 — Speech-to-Text (Whisper TRT)](#7-stage-5--speech-to-text-whisper-trt)
8. [Stage 6 — The Dual-Protocol Orchestrator Connection](#8-stage-6--the-dual-protocol-orchestrator-connection)
9. [Stage 7 — FastAPI Orchestrator Server](#9-stage-7--fastapi-orchestrator-server)
10. [Stage 8 — Hybrid Semantic Router](#10-stage-8--hybrid-semantic-router)
11. [Stage 9 — Agent Dispatch (A2A Protocol)](#11-stage-9--agent-dispatch-a2a-protocol)
12. [Stage 10 — Agent Execution Deep Dive](#12-stage-10--agent-execution-deep-dive)
13. [Stage 11 — LLM Streaming Back Over WebSocket](#13-stage-11--llm-streaming-back-over-websocket)
14. [Stage 12 — TTS (Kokoro ONNX)](#14-stage-12--tts-kokoro-onnx)
15. [Frontend WebSocket Logic](#15-frontend-websocket-logic)
16. [WebSocket Message Type Reference](#16-websocket-message-type-reference)
17. [Thread & Concurrency Architecture](#17-thread--concurrency-architecture)
18. [BUG ANALYSIS: Why Some Agents Produce Multiple Responses But TTS Only Speaks the First](#18-bug-analysis-why-some-agents-produce-multiple-responses-but-tts-only-speaks-the-first)
19. [The Fix](#19-the-fix)

---

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                       NVIDIA JETSON DEVICE                                         │
│                                                                                    │
│  ┌────────────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │     voice_processing/ service      │    │  orchestrator-on-edge/ service      │ │
│  │                                    │    │                                     │ │
│  │  Microphone ──► WakeWord           │    │  FastAPI :8000                      │ │
│  │               ──► VAD              │WS+├─►  /v1/orchestrator/ws/{id}         │ │
│  │               ──► STT              │HTTP│  /v1/orchestrator/message           │ │
│  │               ──► OrchestratorConn│    │  /v1/orchestrator/agent_callback    │ │
│  │                                    │    │                                     │ │
│  │  Speaker  ◄── TTS (Kokoro ONNX)   │◄───│  OrchestratorRouter (semantic)      │ │
│  │                                    │    │  AgentRegistry                      │ │
│  └────────────────────────────────────┘    │  LlmClient (OpenAI / llama.cpp)     │ │
│                                            │                                     │ │
│                                            │  Agents (microservices via A2A):    │ │
│                                            │   car_control :8001                 │ │
│                                            │   reminder    :8002                 │ │
│                                            │   infotainment:8003                 │ │
│                                            │   navigation  :8004                 │ │
│                                            │   user_pref   :8005                 │ │
│                                            │   cloud       :8006                 │ │
│                                            │   ...                               │ │
│                                            └─────────────────────────────────────┘ │
│                                                         │ WS "frontend"             │
│                                            Browser UI ◄─┘                          │
│                                         (React, Vite)                              │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Service Map & Process Boundaries

| Component | File | Language | Runs As |
|---|---|---|---|
| Voice pipeline entry | `voice_processing/main.py` | Python | Standalone process / Docker |
| Wake word | `voice_processing/wake_word.py` | Python + ONNX | Library |
| VAD | `voice_processing/vad.py` | Python + ONNX | Library |
| STT | `voice_processing/stt.py` | Python + Whisper TRT | Library |
| TTS | `voice_processing/tts.py` | Python + Kokoro ONNX | Library |
| Orchestrator WS client | `voice_processing/orchestrator_client.py` | Python | Runs in voice process |
| Orchestrator API server | `orchestrator/main.py` | Python FastAPI | Docker service |
| Routing logic | `orchestrator/router.py` | Python | Imported by API |
| WebSocket manager | `orchestrator/ws_manager.py` | Python | Singleton in FastAPI |
| Agent dispatcher | `orchestrator/client.py` | Python (A2A) | Imported by API |
| LLM client | `orchestrator/llm.py` | Python (AsyncOpenAI) | Imported by API |
| Queue manager (callbacks) | `orchestrator/queue_manager.py` | Python asyncio | Imported by API |
| Car control agent | `agents/car_control/agent_executor.py` | Python | Separate Docker |
| Reminder agent | `agents/reminder/agent_executor.py` | Python | Separate Docker |
| User preference agent | `agents/user_preference/agent_executor.py` | Python | Separate Docker |
| Frontend | `frontend/client/src/components/chat/OrchestratorChat.jsx` | React | Browser |

---

## 3. Stage 1 — Audio Capture (Full-Duplex Stream)

**File:** `voice_processing/main.py`

```python
DEVICE_SAMPLE_RATE = 48000
CHANNELS = 1
CHUNK = 2048

with sd.Stream(
    device=(dev_id, dev_id),       # same device for input AND output
    samplerate=DEVICE_SAMPLE_RATE,
    blocksize=CHUNK,
    channels=CHANNELS,
    callback=audio_callback,
    dtype=np.float32,
):
```

The `sd.Stream` is a **full-duplex** (both directions simultaneously):
- `indata` = microphone capture coming in
- `outdata` = TTS audio going to speaker

### The Callback

```python
def audio_callback(indata, outdata, frames, time_info, status):
    chunk = (indata[:, 0] * 32768).astype(np.int16)  # float32→int16 for processing
    input_buffer.put(chunk)                            # push mic audio to ring buffer

    if not output_queue.empty():
        outdata[:] = output_queue.get_nowait()         # pull TTS chunk from queue
    else:
        outdata.fill(0)                                # silence if nothing to play
```

**Why `* 32768`?**  
sounddevice delivers audio as normalized float32 in [-1.0, 1.0]. All downstream processing (Wake Word, VAD, STT) expects int16 PCM (range [-32768, 32767]). The multiply converts the scale.

**`CircularBuffer`** (`audio_utils.py`) is a FIFO ring buffer with a maximum size of `300 * CHUNK` = 614,400 samples (~12.8 seconds at 48kHz). This prevents unbounded memory growth if the processing thread falls behind.

**`output_queue`** is a standard `queue.Queue`. The TTS processor fills it chunk by chunk; the sounddevice callback drains it in real time.

---

## 4. Stage 2 — Wake Word Detection

**File:** `voice_processing/wake_word.py` — `WakeWordProcessor`

### Models
Three custom ONNX models are loaded simultaneously:
- `hey_dora.onnx`
- `hey_doh_ra.onnx`
- `hey_dola.onnx`

These are loaded through **OpenWakeWord** (`openwakeword.Model`) using the ONNX inference framework.

### Sample Rate Conversion

The device runs at **48kHz** but OpenWakeWord requires **16kHz**. A `Resampler` (from `resampler.py`, which wraps `librosa` or `soxr`) is lazily initialized on first call:

```python
def update_state(self, audio_bytes):
    chunk = np.frombuffer(audio_bytes, dtype=np.int16)
    chunk_f32 = chunk.astype(np.float32) / 32768.0          # int16 → float32
    resampled = self.resampler.resample(chunk_f32)            # 48kHz → 16kHz
    resampled_int16 = (resampled * 32767).astype(np.int16)   # float32 → int16
    self.buffer = np.concatenate((self.buffer, resampled_int16))
```

### Frame-Based Inference

OpenWakeWord processes in **1280-sample frames** (at 16kHz = 80ms windows). Samples are accumulated in `self.buffer`. When enough samples are collected, one frame is sliced off for inference:

```python
to_process = self.buffer[:self._frame_length].copy()
self.buffer = self.buffer[self._frame_length:]
pred = self.model.predict(to_process)
best = max(pred.items(), key=lambda x: x[1])   # highest prob across all models
if prob >= self.detect_prob:   # threshold 0.4 by default
    self._detected = True
```

### One-Shot Detection Flag

`get_detected()` is a **destructive read** — it returns True once and immediately resets the flag:
```python
def get_detected(self):
    out = self._detected
    self._detected = False   # auto-reset
    return out
```

### Debounce

In `process_audio()`, even if `get_detected()` returns True, there's a 0.5-second debounce gate:
```python
if ww.get_detected() and (now - last_wake_word_time) >= WAKE_WORD_DEBOUNCE:
```
This prevents a single loud wake word from triggering twice on the next audio frame.

### What Happens at Wake Word Detection

```python
last_wake_word_time = now
ww_detected = True
timeout_at = now + VAD_FIRST_TIMEOUT   # 3.0s window to start speaking
vad_detected_once = False
last_voice_time = None
pending_turn = False
collected_speech = np.array([], dtype=np.int16)
vad.empty_buffer()                     # clear VAD state
play_notification("listening", ...)    # audio beep feedback

# → PRE-CONNECT to orchestrator WebSocket immediately
with active_conn_lock:
    if active_conn:
        active_conn.close()            # close any previous session
    session_id = f"voice-{uuid.uuid4().hex[:12]}"

    def on_segment(segment):
        tts.speech(segment, language=..., output_queue=..., chunk_size=..., ...)

    active_conn = OrchestratorConnection(session_id=session_id, on_segment=on_segment)
    active_conn.connect()              # opens WS (non-blocking, runs in daemon thread)
```

The key optimization: **WebSocket connection is pre-established during wake word**, not when text is ready. The ~100ms TCP + WS handshake time is hidden behind the time the user takes to say their command.

---

## 5. Stage 3 — Voice Activity Detection (VAD)

**File:** `voice_processing/vad.py` — `VADProcessor`

### Model: Silero VAD

Uses `silero_vad.utils_vad.OnnxWrapper` which is the ONNX export of [Silero VAD](https://github.com/snakers4/silero-vad).

```python
MODEL_CHUNK = 512       # samples per inference call
VAD_SAMPLE_RATE = 16000 # required sample rate

def update_buffer(self, audio_bytes):
    # Resample 48kHz → 16kHz, convert to float32 normalized
    audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    resampled = self.resampler.resample(audio_f32)
    self.buffer = np.concatenate((self.buffer, resampled))

def get_prob(self):
    if len(self.buffer) < MODEL_CHUNK:
        return 0.0
    chunk = self.buffer[:MODEL_CHUNK].copy()
    self.buffer = self.buffer[MODEL_CHUNK:]           # consume processed chunk
    tensor = torch.from_numpy(chunk).float()
    out = self.model(tensor, VAD_SAMPLE_RATE)
    return out.item()                                  # float in [0.0, 1.0]
```

At 16kHz, 512 samples = **32ms per inference**. This gives 31 detections per second.

### VAD in the Processing Loop

```python
if ww_detected:
    vad.update_buffer(chunk.tobytes())              # feed audio to VAD buffer
    collected_speech = np.concatenate((collected_speech, chunk))  # accumulate raw audio

p = vad.get_prob()
if p > 0.5:                    # speech detected threshold
    vad_detected_once = True
    last_voice_time = now      # update timestamp of last detected speech
```

`collected_speech` is the raw 48kHz int16 audio that will be sent to STT. VAD just tells us *when* to stop collecting.

---

## 6. Stage 4 — STT Timing State Machine

**File:** `voice_processing/main.py` — inside `process_audio()` loop

This is the most subtle part of the pipeline. The state machine decides **when to cut audio segments and send them to STT**.

### Variables

| Variable | Type | Meaning |
|---|---|---|
| `ww_detected` | bool | Wake word triggered, currently in listening mode |
| `vad_detected_once` | bool | At least one VAD > 0.5 event happened |
| `last_voice_time` | float | Timestamp of last VAD detection |
| `pending_turn` | bool | At least one segment has been sent to STT |
| `timeout_at` | float | Deadline for user to start speaking (3.0s) |
| `collected_speech` | np.array | Raw audio accumulated since last segment flush |
| `SEGMENT_SILENCE` | 0.7s | Silence after which a segment is cut (mid-turn) |
| `TURN_END_SILENCE` | 1.5s | Silence after which the full turn is ended |

### The State Machine (executed every ~42ms per chunk)

```
After wake word detected:
  timeout_at = now + 3.0s

  For each audio chunk:
    ┌─ Accumulate into collected_speech
    ├─ Feed to VAD → get prob p
    │
    ├─ If p > 0.5:
    │    last_voice_time = now
    │    vad_detected_once = True
    │
    └─ Check timing:

       IF last_voice_time is set:
         segment_at  = last_voice_time + 0.7s   (SEGMENT_SILENCE)
         turn_end_at = last_voice_time + 1.5s   (TURN_END_SILENCE)

         CASE A: now >= turn_end_at AND pending_turn is True
           → stt_queue.put(END_TURN)            # signal full turn done
           → play "sending" notification
           → reset all state

         CASE B: now >= segment_at AND collected_speech not empty
           → if len(collected_speech)/48000 >= 0.095s  (MIN_SEGMENT_SEC)
               stt_queue.put(collected_speech.copy())  # partial STT
               pending_turn = True
           → collected_speech = []

         CASE C: now >= turn_end_at AND NOT pending_turn
           → timeout (user started but stopped too quickly)
           → play "timeout"
           → reset

       ELSE IF timeout_at is set AND now >= timeout_at:
         → timeout (user never spoke)
         → play "timeout"
         → reset
```

**Why the two-tier silence design?**

- `SEGMENT_SILENCE = 0.7s`: Triggers partial STT while the user might still be speaking. This allows STT to warm up and check partial text during pauses.
- `TURN_END_SILENCE = 1.5s`: Only after a pending segment was already sent, longer silence = "user is done talking."

This means STT can run **during** the turn — overlapping with the user's pauses — rather than waiting for the full utterance.

---

## 7. Stage 5 — Speech-to-Text (Whisper TRT)

**File:** `voice_processing/stt.py` — `STTProcessor`

### Model Loading

```python
trt_path = os.environ.get("WHISPER_TRT_MODEL_PATH")
if trt_path:
    # Use local TRT engine file
    self.model = load_trt_model("small", path=trt_path, build=False)
else:
    # Use cached TRT (built on first run, cached to disk)
    set_cache_dir(_CACHE_DIR)
    self.model = load_trt_model("small", path=None, build=False)

# Warmup pass (required to initialize TRT engine)
self.model.transcribe(np.zeros(1536, dtype=np.float32), language=language)
```

`whisper_trt` is a TensorRT-optimized Whisper. On Jetson it runs the encoder/decoder on GPU via TRT engines for 5–10× speedup over PyTorch Whisper.

### Audio Resampling

```python
def resample_audio(audio_int16, original_sr, target_sr):
    padded = _pad_audio(audio_int16.tobytes(), 2, 1)  # pad to sample boundary
    seg = AudioSegment(data=padded, sample_width=2, frame_rate=original_sr, channels=1)
    seg = seg.set_frame_rate(target_sr)
    return np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
```

Uses `pydub` (which calls `ffmpeg` internally) for high-quality resampling. Falls back to linear interpolation via `np.interp` if `pydub` fails.

### Transcription

```python
def transcribe(self, audio_int16):
    samples = resample_audio(audio_int16, self.device_sample_rate, 16000)
    result = self.model.transcribe(samples, language=self.language)
    
    text = result.get("text", "")
    detected_lang = result.get("language")
    
    # Fallback: parse lang tags like <|en|> from text
    if not detected_lang:
        m = re.search(r"<\|([a-z]{2})\|>", text)
        if m: detected_lang = m.group(1)
    
    # Clean Whisper special tokens: <|startoftranscript|>, <|en|>, etc.
    clean = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return {"text": clean, "language": detected_lang}
```

### STT Processing Thread (`process_stt()`)

```python
allowed_langs = ["vi", "en", "ja", "ko"]
turn_texts = []      # accumulates segment texts
turn_lang = None     # detected language of the turn

while not stop.is_set():
    item = stt_queue.get(timeout=1.0)

    if item is END_TURN:
        # ── COMPLETE TURN: merge all segments ──
        text = " ".join(turn_texts).strip()
        turn_texts = []
        if not text: continue
        lang = turn_lang

        # → send to orchestrator
        with active_conn_lock:
            if active_conn:
                active_conn.send_message(text)        # HTTP POST
            else:
                send_and_stream(text, session_id, ...)  # fallback

    else:
        # ── SEGMENT: transcribe and accumulate ──
        out = stt.transcribe(item)
        part = out.get("text", "").strip()
        if part:
            turn_texts.append(part)
        if out.get("language") in allowed_langs:
            turn_lang = out.get("language")
        continue    # ← don't send yet, wait for END_TURN
```

**Key insight:** Multiple partial segments are transcribed progressively as the user pauses mid-sentence. The texts are accumulated in `turn_texts`. Only when `END_TURN` arrives (after 1.5s total silence + pending turn) are all segments joined and sent to the orchestrator.

---

## 8. Stage 6 — The Dual-Protocol Orchestrator Connection

**File:** `voice_processing/orchestrator_client.py` — `OrchestratorConnection`

### Why Two Protocols?

The system uses **WebSocket for receiving** and **HTTP POST for sending** — deliberately decoupled:

```
voice_processing                             orchestrator (FastAPI)
      │                                              │
      │─── WS CONNECT /ws/voice-abc123 ────────────▶│  registered in ws_manager
      │    (daemon thread, persistent)               │
      │                                              │
      │─── HTTP POST /message ──────────────────────▶│  triggers process_request()
      │◀─── 202 Accepted (immediate) ───────────────│  (returns immediately)
      │                                              │
      │◀─── WS: {"type": "token", "content": "Hi"} ─│  LLM token
      │◀─── WS: {"type": "token", "content": " !"} ─│  LLM token
      │◀─── WS: {"type": "done", "message": "..."} ─│  LLM finished
      │                                              │
      │ (agent callback later)                        │
      │◀─── WS: {"type": "done", "message": "Added"} │  ← from queue_manager
```

This decoupling enables:
1. **Non-blocking TTS** — voice client can start TTS on first sentence while LLM generates the rest
2. **Agent callbacks** — agents that have async results (reminders, preferences) can push them later over the same WS

### Connection Lifecycle

```python
class OrchestratorConnection:
    def __init__(self, base_url=None, session_id=None, on_segment=None):
        self.base_url = base_url or os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
        self.session_id = session_id
        self.on_segment = on_segment    # callback → what to do with each TTS segment
        self.ws = None
        self.ws_thread = None
        self.buffer = []               # token accumulation buffer
        self.SENTENCE_END = (".", "!", "?")
        self.is_done = threading.Event()

    def connect(self):
        ws_url = f"ws://...{self.base_url}/v1/orchestrator/ws/{self.session_id}"
        self.ws = websocket.WebSocketApp(ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        # Non-blocking: runs ws.run_forever() in a daemon thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever,
                                          kwargs={"ping_interval": 20},
                                          daemon=True)
        self.ws_thread.start()
```

### Token Buffering and Sentence-Boundary TTS Splitting

```python
def on_message(ws, raw):
    d = json.loads(raw)
    t = d.get("type", "")

    if t == "token":
        content = d.get("content", "")
        if content:
            self.buffer.append(content)
            text = "".join(self.buffer).rstrip()
            if text.endswith(self.SENTENCE_END):   # ".", "!", "?"
                self.flush()                        # → TTS this sentence now

    elif t in ("done", "error"):
        self.flush()          # flush any remaining buffered tokens
        self.is_done.set()    # signal that the WS session is complete
```

```python
def flush(self):
    if not self.buffer: return
    segment = "".join(self.buffer).strip()
    self.buffer.clear()
    if segment and self.on_segment:
        self.on_segment(segment)    # → TTSProcessor.speech()
```

**Why sentence-boundary splitting?**  
TTS synthesis (Kokoro ONNX) takes 0.3–1.5 seconds per call. If TTS had to wait for the full LLM response before starting, the user would wait for ALL of: LLM generation time + TTS synthesis time. By splitting at sentence boundaries, the user starts hearing "Hello, I've opened the door." while the LLM is still generating "Would you like me to do anything else?"

### Sending the Message

```python
def send_message(self, user_message, post_timeout=30):
    post_url = f"{self.base_url}/v1/orchestrator/message"
    r = requests.post(post_url,
        json={"message": user_message, "session_id": self.session_id},
        timeout=post_timeout,
        headers={"Content-Type": "application/json"}
    )
    r.raise_for_status()   # raises if HTTP 4xx/5xx
    return True
```

The HTTP POST returns almost immediately (`202 Accepted`). The actual processing — routing, agent calls, LLM streaming — all happens asynchronously via `asyncio.create_task()` on the server side.

---

## 9. Stage 7 — FastAPI Orchestrator Server

**File:** `orchestrator/main.py` + `orchestrator/api/route_orchestrator.py`

### App Setup

```python
app = FastAPI(title="Edge Orchestrator API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
app.include_router(agents_router,       prefix="/v1")  # /v1/agents
app.include_router(orchestrator_router, prefix="/v1")  # /v1/orchestrator
app.include_router(favorites_router,    prefix="/v1")  # /v1/favorites
```

### Global Singletons (initialized at module import time)

```python
registry = AgentRegistry()               # loads agents from config/agent_list.json
context_store = PerAgentContext()        # per-session conversation history
orchestrator_router = OrchestratorRouter()  # loads SentenceTransformer + BM25
ws_manager = WebSocketManager()          # dict: session_id → WebSocket
llm_client = LlmClient()                 # AsyncOpenAI client
favorite_store = FavoritesStore()        # favorites JSON store
session_locks = {}                       # per-session asyncio.Lock
```

### POST /v1/orchestrator/message

```python
@router.post("/message", response_model=MessageAccepted)
async def post_message(req: MessageRequest):
    user_message = req.message.strip()
    session_id   = req.session_id or "default"

    # 1. Route to agents (synchronous, fast)
    routing_result = await orchestrator_router.route(user_message)
    agent_ids = routing_result["agent_ids"]

    if not agent_ids:
        raise HTTPException(404, "No agent available")

    # 2. Store context
    context_store.add(session_id, agent_ids[0], "user", user_message)

    # 3. Fire and forget processing in background
    asyncio.create_task(process_request(session_id, routing_result))

    # 4. Return immediately
    return MessageAccepted(accepted=True, session_id=session_id, agent_ids=agent_ids)
```

**Timing:** The whole handler runs in ~50–200ms (routing + context store). The actual work (agents + LLM) runs asynchronously.

### WebSocket /v1/orchestrator/ws/{session_id}

```python
@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_manager.connect(session_id, websocket)   # register: dict[session_id] = ws
    try:
        while True:
            await websocket.receive_text()             # keep-alive loop
    except Exception:
        ws_manager.disconnect(session_id)             # deregister on disconnect
```

The WS endpoint just registers the connection and maintains it alive. All writes to the socket happen from other coroutines via `ws_manager.send(session_id, message)`.

### WebSocketManager

```python
class WebSocketManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}    # session_id → WebSocket

    async def connect(self, session_id, websocket):
        await websocket.accept()
        self.connections[session_id] = websocket

    async def send(self, session_id, message: dict):
        ws = self.connections.get(session_id)
        if ws:
            await ws.send_json(message)               # FastAPI serializes dict → JSON

    async def broadcast(self, message: dict):
        for ws in list(self.connections.values()):
            await ws.send_json(message)
```

### The Frontend/Voice Bridge

```python
async def _send_to_session_and_frontend(session_id: str, message: dict):
    await ws_manager.send(session_id, message)       # → voice client

    if session_id.startswith("voice-"):
        await ws_manager.send("frontend", {           # → browser UI
            **message,
            "session_id": session_id                  # attach origin session
        })
```

This is the **bridge** that makes voice conversations visible in the browser. The browser always connects as `"frontend"` (hardcoded). When a voice session (`voice-{uuid}`) generates tokens, BOTH the voice client AND the browser receive the same messages (with `session_id` added for the browser to know which voice session it belongs to).

---

## 10. Stage 8 — Hybrid Semantic Router

**File:** `orchestrator/router.py` — `OrchestratorRouter`

### Startup: Building the Corpus

At init time:
1. `AgentRegistry.get_all()` — fetches each agent's `/.well-known/agent-card.json` via HTTP
2. Each agent card contains: `name`, `description`, `skills[].description`, `skills[].examples[]`
3. All this text is concatenated into a single document per agent
4. `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")` encodes all documents → dense vectors
5. `BM25Okapi` is built on tokenized corpus for keyword matching

### Translation Layer

```python
translator = SimpleArgosTranslator()
translated_full, src_lang = translate_cached(user_message)
```

Non-English messages are translated to English before embedding. This is because the agent corpus is in English. The **original** (untranslated) text is preserved for display and for agent processing that supports multiple languages.

### Multi-Intent Parsing

```python
# "Open the trunk and play some rock music"
parts = self.mult_intent_parser.parse(user_message)
# → ["Open the trunk", "play some rock music"]
```

`MultiIntentParser` splits on conjunction words (`and`, `then`, `그리고`, `고`, `て`) using language-aware regex. Korean and Japanese use different grammatical connectives, so the parser detects the script first via Unicode ranges:

```python
_RE_HANGUL   = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")
_RE_HIRAGANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
```

After splitting, fragments that are "too short" and have no verb (checked via spaCy POS tagger) are merged back with the previous fragment to avoid splitting "open the door" into ["open the", "door"].

### Hybrid Scoring (route_one)

```python
def hybrid_scores(self, text: str, q_emb):
    emb  = embedding_scores(q_emb)    # cosine similarity: 60% weight
    bm25 = bm25_scores(text)           # BM25 keyword: 40% weight
    if bm25.max() > 0:
        bm25 = bm25 / bm25.max()      # normalize BM25 to [0,1]
    return 0.6 * emb + 0.4 * bm25
```

**Delta check** (avoids fuzzy multi-selection when one agent clearly wins):
```python
if best_score - second_score > 0.2:    # clear winner
    return [best_agent_id]             # only return one agent
# else: threshold filtering applies
```

### Route Result Structure

```python
routing_result = {
    "message": "Open the trunk and play some rock music",
    "translated_text": "Open the trunk and play some rock music",
    "multi": True,
    "agent_ids": ["car_control", "infotainment"],
    "parts": [
        {
            "text": "Open the trunk",
            "translated_text": "Open the trunk",
            "selected": ["car_control"],
            "scores": {"car_control": 0.87, "infotainment": 0.21}
        },
        {
            "text": "play some rock music",
            "translated_text": "play some rock music",
            "selected": ["infotainment"],
            "scores": {"infotainment": 0.91, "car_control": 0.15}
        }
    ]
}
```

---

## 11. Stage 9 — Agent Dispatch (A2A Protocol)

**File:** `orchestrator/client.py`

### Batch Building

Each agent receives only the sub-intents that were routed to it:
```python
def build_agent_batches(routing_result):
    # car_control → ["Open the trunk"]
    # infotainment → ["play some rock music"]
    batches = {}
    for part in routing_result["parts"]:
        for aid in part["selected"]:
            batches.setdefault(aid, []).append(part["text"])
    return batches
```

### Parallel Dispatch via asyncio.gather

```python
coros = [
    send_to_agent(agent_id, user_messages, context, session_id, registry, metadata)
    for agent_id, user_messages in batches.items()
]
results = await asyncio.gather(*coros, return_exceptions=True)
```

All agents are called **concurrently** — a multi-intent command like "open door and play music" has both agents running simultaneously.

### A2A Message Construction

```python
req = SendMessageRequest(
    id=uuid4().hex,
    params=MessageSendParams(
        message={
            "role": "user",
            "parts": [
                {"type": "text", "text": "Open the trunk"},  # one part per sub-intent
            ],
            "messageId": uuid4().hex,
            "metadata": {
                "session_id": session_id,
                "agent_id": agent_id,
                "routing_result": routing_result,   # ← full routing context passed!
                "favorites": favorite_store.get_all(),
                "translated_text": routing_result.get("translated_text"),
            }
        }
    )
)
```

The `routing_result` is passed through the metadata so agents can look up which sub-intent text was specifically routed to them (used by `car_control` to extract the translated sub-intent text).

### Response Extraction

Agent responses are JSON strings embedded inside A2A protocol parts:
```python
def _extract_dict_from_response(response):
    for part in result.parts:
        if part.root.kind == "text":
            return json.loads(part.root.text)   # e.g. {"success": true, "message": [...]}
```

---

## 12. Stage 10 — Agent Execution Deep Dive

### Car Control Agent (`agents/car_control/agent_executor.py`)

**Two-stage NLI classification:**

1. **BM25 pre-filter** — rapidly ranks candidate action labels by keyword overlap, keeps top-K:
   ```python
   filtered_labels = self.filter_candidates(text, top_k=3)
   # "open the trunk" → ["open trunk", "close trunk", "open left door"]
   ```

2. **Zero-shot NLI** — runs `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` on reduced label set:
   ```python
   results = self.classifier([text], filtered_labels, multi_label=True)[0]
   actions = [label for label, score in zip(results["labels"], results["scores"])
              if score >= self.threshold]   # 0.5 default
   ```

**Conflict resolution:**
```python
CONFLICT_GROUPS = [
    {"open left door", "close left door"},
    {"open right door", "close right door"},
    ...
]
# If both "open left door" and "close left door" detected → keep the first
```

**Output:**
```json
{
  "success": true,
  "message": [{"action": "open trunk", "success": true}],
  "data": {"action": ["open_trunk"]}
}
```

### Reminder & User Preference Agents — The ACK Pattern

These agents follow an **ACK + async background** pattern:

```python
async def execute(self, context, event_queue):
    text = _get_user_text(context)
    session_id = metadata.get("session_id")

    # 1. ACK immediately (synchronous HTTP response returns now)
    await self._safe_send(event_queue, True, "Reminder request received and processing.")

    # 2. Heavy work in background
    asyncio.create_task(self._process(text, session_id, event_queue))
```

The `execute()` method returns quickly with an acknowledgment. The actual LLM-based extraction and storage happens in `_process()` asynchronously.

**`_process()` then does TWO things:**

```python
# 1. Send result to event_queue (this goes back through A2A to sync response)
await self._safe_send(event_queue, True, result_msg)

# 2. HTTP POST callback to orchestrator (async delayed response)
await loop.run_in_executor(None, self.post_callback, session_id, result_msg, agent_id)
```

This results in **two separate messages from the agent**: the ACK and the actual result. The orchestrator receives both.

---

## 13. Stage 11 — LLM Streaming Back Over WebSocket

**File:** `orchestrator/api/route_orchestrator.py` — `process_request()` + `stream_llm()`

### Full Processing Flow

```python
async def process_request(session_id, routing_result):
    user_message = routing_result["message"]
    agent_ids    = routing_result["agent_ids"]

    # Step 0: notify frontend of user message (for voice sessions)
    if session_id.startswith("voice-"):
        await ws_manager.send("frontend", {
            "type": "user_message",
            "session_id": session_id,
            "content": user_message,
            "agent_ids": agent_ids
        })

    # Step 1: call all agents concurrently
    agent_response = await send_to_agents_parallel(
        user_message, context_store, session_id, registry, routing_result,
        metadata={"favorites": ..., "translated_text": ..., "routing_result": routing_result}
    )

    # Step 2: post-process (collect URLs, images, car actions)
    final_agent_response = post_process_agent_response(agent_response)

    # Step 3: push raw agent data to client (BEFORE LLM!) for immediate UI/3D model update
    await _send_to_session_and_frontend(session_id, {
        "type": "data",
        "content": final_agent_response
    })

    # Step 4: build LLM prompt from agent results
    messages = [
        {"role": "system", "content": ORCHESTRATOR_PROMPT},
        {"role": "user", "content":
            f"This is the user request:\n{user_message}\n\n"
            f"This is the agent_response:\n{final_agent_response['responses']}"
        }
    ]

    # Step 5: stream LLM (tokens → WebSocket)
    await stream_llm(session_id, messages)
```

### stream_llm — Token-by-Token Push

```python
async def stream_llm(session_id, messages):
    lock = get_session_lock(session_id)

    async with lock:   # ← BLOCKS concurrent LLM calls for same session
        full_text = ""
        async for token in llm_client.stream_chat_completion(messages):
            full_text += token
            await _send_to_session_and_frontend(session_id, {
                "type": "token",
                "content": token   # 1–5 characters typically
            })

        await _send_to_session_and_frontend(session_id, {
            "type": "done",
            "message": full_text   # full accumulated text
        })
```

**Session Lock:** The `asyncio.Lock` per `session_id` ensures that if an agent callback arrives while LLM is still streaming, it WAITS until LLM is completely done before sending. This prevents the agent callback's "done" message from interleaving with the LLM token stream.

### LLM Client

```python
class LlmClient:
    async def stream_chat_completion(self, messages):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,  # 256 default (short voice-friendly responses)
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content   # yield one token at a time
```

The `AsyncOpenAI` client supports both:
- **OpenAI API** — remote, full models
- **llama.cpp** — local, OpenAI-compatible server (`http://localhost:8080/v1`)

---

## 14. Stage 12 — TTS (Kokoro ONNX)

**File:** `voice_processing/tts.py` — `TTSProcessor`

### Model

Kokoro is a fast, multilingual TTS model. The ONNX export runs efficiently on Jetson GPU/CPU.

```python
kokoro = Kokoro(onnx_path, voices_path)
```

`onnx_path` = `kokoro-v1.0.onnx` (model weights)  
`voices_path` = `voices-v1.0.bin` (speaker embeddings — "af_heart" voice)

### Language Mapping

```python
LANGUAGE_MAP = {
    "en": "en-us", "ja": "ja", "vi": "vi", "zh": "zh",
    "fr": "fr", "es": "es", "de": "de", "it": "it",
    "pt": "pt", "ru": "ru", "ko": "ko", "hi": "hi",
}
```

Language is detected by STT (Whisper) and passed all the way through to TTS.

### Audio Generation & Playback

```python
def speech(self, text, language=None, output_queue=None, chunk_size=None, device_sample_rate=None):
    lang = self._map_lang(language)
    audio, sr = self.kokoro.create(
        text=text.strip(),
        voice="af_heart",
        speed=1.0,
        lang=lang,
        trim=True,        # trim leading/trailing silence
    )
    # audio: np.float32 array, sr: 24000

    # Resample from Kokoro's 24kHz → device's 48kHz
    if sr != device_sample_rate:
        r = Resampler(sr, device_sample_rate)
        audio = r.resample(audio)

    # Chunk into CHUNK=2048 sample frames and push to output_queue
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))   # zero-pad last chunk
        output_queue.put(chunk.reshape(chunk_size, 1).astype(np.float32))
```

The `output_queue.put()` is blocking — it will wait if the queue is full. The `audio_callback` in sounddevice drains the queue in real time at 2048 samples / 48000Hz = **42.6ms per chunk**.

---

## 15. Frontend WebSocket Logic

**File:** `frontend/client/src/components/chat/OrchestratorChat.jsx`

### Connection Setup

```javascript
const FRONTEND_SESSION_ID = "frontend"
const WS_URL = getOrchestratorWsUrl()  // from urlUtils, dynamic based on window.location

useEffect(() => {
    const ws = new WebSocket(`${WS_URL}/${FRONTEND_SESSION_ID}`)
    ws.onmessage = (event) => { /* ... */ }
    wsRef.current = ws
    return () => ws.close()
}, [])  // ← empty deps: runs once on mount, never reconnects
```

### Message Handler

```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    const isVoice = data.session_id?.startsWith("voice-")

    // Voice session tracking maps (in-memory, ref-based to avoid re-renders)
    // voiceSessionToMsgId: sid → assistant bubble ID
    // voiceSessionToUserMsgId: sid → user bubble ID
    // voiceSessionToAgentIds: sid → agent_ids array

    if (data.type === "user_message" && isVoice) {
        // Create user bubble showing what was heard
        setChatHistory(prev => [...prev, {
            id: userMsgId, from: "User", to: "Orchestrator", content: data.content
        }])
        return
    }

    if (data.type === "data") {
        // Dispatch to 3D model: window.dispatchEvent(new CustomEvent('CAR_CONTROL_ACTION', ...))
        // Update agent data payload in chat bubble
    }

    if (data.type === "token") {
        // Append token to streaming bubble content
    }

    if (data.type === "done") {
        // Mark bubble as complete, setIsDone(true), setActiveAgents([])
    }
}
```

### Text Input (Manual Mode — `/test` route only)

```javascript
const sendMessage = async () => {
    const payload = { message: text, session_id: sessionId.current }  // "frontend"

    // 1. Show user bubble immediately
    setChatHistory(prev => [...prev, { from: "User", content: text }])

    // 2. HTTP POST to trigger processing
    const res = await fetch(API_URL, { method: "POST", body: JSON.stringify(payload) })
    const data = await res.json()    // { accepted: true, agent_ids: [...] }

    // 3. Pre-create streaming assistant bubble
    currentMsgId.current = nextId
    setChatHistory(prev => [...prev, {
        id: nextId, from: "Orchestrator", to: data.agent_ids, content: "", streaming: true
    }])
}
```

### 3D Model Bridge

```javascript
if (data.content) {
    window.dispatchEvent(new CustomEvent('CAR_CONTROL_ACTION', {
        detail: data.content
    }))
}
```

`CarModelView.jsx` listens for `'CAR_CONTROL_ACTION'` on `window`. The `detail` will be:
```json
{
    "responses": [{"agent_id": "car_control", "success": true, "message": [...]}],
    "data": {
        "action": ["open_door_right"],
        "urls": [],
        "images": []
    }
}
```
The component maps `action` strings like `"open_door_right"` to 3D animation triggers.

---

## 16. WebSocket Message Type Reference

| `type` | Direction | Key Fields | Purpose |
|---|---|---|---|
| `user_message` | Server → Frontend | `session_id`, `content`, `agent_ids` | Show user turn bubble for voice sessions |
| `data` | Server → Client | `content: {responses, data}` | Agent results payload (triggers 3D animations) |
| `token` | Server → Client | `content: "word"` | One streaming LLM token |
| `done` | Server → Client | `message: "full text"` | LLM stream complete OR agent callback result |
| `error` | Server → Client | `message: "error desc"` | A processing error occurred |
| `session_log` | Server → Frontend | `content: "log line"` | Internal log forwarding for debug UI |
| `backend_log` | Server → Client | `content: "log line"` | Per-message execution log (shown in details modal) |

---

## 17. Thread & Concurrency Architecture

```
┌─── sounddevice callback thread (OS audio thread) ──────────────────┐
│  audio_callback() runs every 42.6ms                                 │
│    input_buffer.put(mic_chunk)                                      │
│    outdata = output_queue.get_nowait() or silence                   │
└─────────────────────────────────────────────────────────────────────┘
           │ (CircularBuffer)                ▲ (queue.Queue)
           ▼                                │
┌─── Thread: process_audio() ─────────────────────────────────────────┐
│  while True:                                                         │
│    chunk = input_buffer.get(CHUNK)                                   │
│    ww.update_state(chunk) → check detection                          │
│    if ww_detected: vad.update_buffer(chunk)                          │
│    p = vad.get_prob()                                                │
│    [state machine: cut segments, set timing flags]                   │
│    stt_queue.put(audio_chunk)  or  stt_queue.put(END_TURN)         │
└─────────────────────────────────────────────────────────────────────┘
           │ (queue.Queue, maxsize=5)
           ▼
┌─── Thread: process_stt() ───────────────────────────────────────────┐
│  while True:                                                         │
│    item = stt_queue.get()                                            │
│    if item is END_TURN: → active_conn.send_message(text)  [HTTP]    │
│    else: stt.transcribe(audio) → accumulate turn_texts              │
└─────────────────────────────────────────────────────────────────────┘
           │ (HTTP POST)
           ▼
┌─── OrchestratorConnection WS Thread (daemon) ───────────────────────┐
│  ws.run_forever()                                                    │
│  on_message: buffer tokens, flush on sentence boundary              │
│  on flush: → on_segment() → tts.speech()                           │
└─────────────────────────────────────────────────────────────────────┘
                                                    ▼ (fills output_queue)
                                                ← sounddevice plays

┌─── FastAPI uvicorn (asyncio event loop) ────────────────────────────┐
│  POST /message → asyncio.create_task(process_request)               │
│  process_request:                                                    │
│    await send_to_agents_parallel     (concurrent HTTP)               │
│    await stream_llm (async with lock)                                │
│      async for token in llm: ws_manager.send(session_id, token)     │
│  WS /ws/{id}: keep-alive receive loop                               │
│  POST /agent_callback: asyncio.create_task(delayed_send)            │
│    delayed_send: async with lock: enqueue_agent_callback            │
│    queue_manager worker: ws_manager.send(session_id, done_msg)      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 18. BUG ANALYSIS: Why Some Agents Produce Multiple Responses But TTS Only Speaks the First

### The Agents That Do This

Both `reminder` and `user_preference` agents use an **ACK + async background** pattern. They produce **two** separate event_queue messages during a single orchestrator request:

```python
# In execute() — synchronous, fast:
await self._safe_send(event_queue, True, "Reminder request received and processing.")
asyncio.create_task(self._process(...))

# In _process() — async, slow (LLM extraction + storage):
await self._safe_send(event_queue, True, result_msg)           # "Added reminder 'dentist' successfully"
self.post_callback(session_id, result_msg, self.agent_id)       # HTTP POST to /agent_callback
```

So the `send_to_agents_parallel` call receives **the ACK** ("Reminder request received and processing.") as the synchronous A2A response. The **actual result** ("Added reminder...") comes asynchronously via `agent_callback`.

### Response Flow Timeline

```
T+0ms:   HTTP POST /v1/orchestrator/message received
T+50ms:  orchestrator_router.route() → reminder agent
T+100ms: send_to_agents_parallel → HTTP to reminder agent
             reminder.execute() starts:
               → ACK enqueued: "Reminder request received and processing."
               → asyncio.create_task(_process())   # fire and forget
             → A2A response returned with ACK text

T+200ms: process_request() gets agent_response = [{"message": "Reminder request received..."}]
         → post_process → final_agent_response
         → _send_to_session_and_frontend({"type":"data", ...})   # DATA sent to WS
         → stream_llm() starts, acquires lock
             LLM generates reply using ACK as context
             → "token" events streamed to WS
         → "done" event sent to WS
         → lock released

T+300ms: (background) _process() finishes:
           "Added reminder 'dentist' successfully"
           → event_queue.enqueue_event(...)  ← this is IGNORED by A2A after response returned
           → self.post_callback(session_id, "Added reminder 'dentist'...", agent_id)

T+310ms: orchestrator receives POST /agent_callback
           → delayed_send: async with lock (LOCKS AFTER LLM IS DONE) → enqueue_agent_callback
           → queue_manager worker: ws_manager.send(session_id, {
               "type": "done",
               "message": "Added reminder 'dentist' successfully",
               "source": "reminder"
             })
```

### The Bug: What the Voice Client Does with Callback "done"

The voice client `OrchestratorConnection.on_message` only does this:

```python
elif t in ("done", "error"):
    self.flush()           # flush any buffered tokens
    self.is_done.set()     # ← EVENT FIRED, session treated as complete
```

When the **LLM's** `done` event arrives:
- `flush()` flushes remaining buffered LLM tokens → TTS speaks them ✅
- `is_done.set()` signals completion

When the **agent callback's** `done` event arrives (later):
- `flush()` is called but `self.buffer` is **empty** (no tokens were buffered for this message)
- The text in `d["message"]` = `"Added reminder 'dentist' successfully"` is **NEVER READ**
- `self.is_done.set()` again (already set, no-op)
- The callback result text is **completely silently dropped** — TTS never gets it

### Root Cause — One Line

```python
# orchestrator_client.py — on_message handler
elif t in ("done", "error"):
    self.flush()           # only flushes token buffer
    self.is_done.set()     # ← treats ANY "done" as "we're done"

# The "message" field of a callback "done" is NEVER READ:
# d.get("message") → "Added reminder 'dentist' successfully" ← DROPPED
```

The handler assumes `done` always means "the LLM is done streaming tokens." It does not account for agent callback `done` messages, which carry the actual text in `d["message"]` rather than in preceding `token` events.

### Why Only First Response is Spoken

```
LLM stream:       token token token token → done["message"="..."]
                  ↑ these go to buffer → sentence flush → TTS ✅

Agent callback:   (no tokens) → done["message"="Added reminder..."]
                  ↑ no buffer to flush → message dropped → NO TTS ❌
```

The LLM response **is** spoken because its content arrives as individual `token` events that are buffered and flushed at sentence boundaries. The agent callback response is **not** spoken because it arrives as a single `done` event with text in the `message` field — and `on_message` never reads that field.

---

## 19. The Fix

There are two clean solutions depending on how you want the behavior.

### Option A — Quick Fix: Read `message` Field from Callback Done Events

Modify `OrchestratorConnection.on_message` in `voice_processing/orchestrator_client.py` to treat a `done` event with a `message` field (and empty internal buffer) as a new TTS segment:

```python
# CURRENT (buggy):
elif t in ("done", "error"):
    self.flush()
    self.is_done.set()
```

```python
# FIXED:
elif t in ("done", "error"):
    if self.buffer:
        # Normal case: LLM stream ended, flush remaining buffered tokens
        self.flush()
    else:
        # Agent callback case: no preceding tokens, text is in "message" field
        callback_msg = d.get("message", "").strip()
        if callback_msg and self.on_segment:
            # Only speak if this is NOT just the LLM's final done message
            # (LLM done messages have empty buffer too if last token was sentence-end)
            # Distinguish by checking "source" field added by queue_manager
            source = d.get("source", "")
            if source and source not in ("", None):
                # This is an agent callback → speak it
                self.on_segment(callback_msg)

    self.is_done.set()
```

> **Gotcha:** The LLM's `done` event also has an empty buffer if the last token ended in `.!?` (it was already flushed). To distinguish LLM done from callback done, you can check the `source` field — `queue_manager` sets `"source": agent_id` in callback messages, while the LLM's done event has no `source` field.

### Option B — Cleaner Fix: Add a New Message Type for Callbacks

Change `queue_manager.py` on the server side to send a distinct type instead of reusing `"done"`:

```python
# In orchestrator/queue_manager.py — fire_and_forget_worker:
await send_fn(session_id, {
    "type": "agent_result",      # ← NEW TYPE, distinct from "done"
    "message": payload["message"],
    "source": payload.get("agent_id", "agent"),
    "ts": payload.get("ts"),
})
```

Then handle it explicitly in the voice client:

```python
# In voice_processing/orchestrator_client.py — on_message:
elif t == "agent_result":
    msg = d.get("message", "").strip()
    if msg and self.on_segment:
        self.on_segment(msg)      # → TTS speaks the callback result

elif t in ("done", "error"):
    self.flush()
    self.is_done.set()
```

And in the frontend `OrchestratorChat.jsx` — add handling for `agent_result`:

```javascript
if (data.type === "agent_result") {
    // Append agent result as a new message bubble
    setChatHistory(prev => [...prev, {
        id: uuid(),
        from: data.source || "Orchestrator",
        to: [],
        content: data.message || "",
        streaming: false
    }])
}
```

### Option B is Recommended Because

1. **Semantic clarity** — `done` means "LLM is done". `agent_result` means "agent finished its async work". Two different things should have different names.
2. **No fragile heuristics** — no need to check `source` field or buffer emptiness to distinguish.
3. **Frontend and voice client both benefit** — they can both handle `agent_result` cleanly.
4. **Future-proof** — if more agents add async callbacks, they all work automatically.

### Files to Modify for Option B

| File | Change |
|---|---|
| `orchestrator/queue_manager.py` | Change `"type": "done"` → `"type": "agent_result"` in `fire_and_forget_worker` |
| `voice_processing/orchestrator_client.py` | Add `elif t == "agent_result": self.on_segment(d.get("message",""))` |
| `frontend/.../OrchestratorChat.jsx` | Add `data.type === "agent_result"` handler to create a new chat bubble |

---

*Generated: 2026-04-14 | System: Edge AI Orchestration on NVIDIA Jetson*
