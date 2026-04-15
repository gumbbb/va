# Orchestrator on Edge: Complete System & Architecture Deep Dive

An AI-powered in-car assistant demo that runs entirely on edge devices (such as NVIDIA Jetson). The system uses a multi-agent architecture where a central orchestrator routes user requests to specialized agents, each handling a different domain (vehicle control, manual lookup, navigation, infotainment, cloud search).

> This README is an **in-depth, end-to-end technical breakdown** of the system, designed to give you a very deep understanding of the architecture, pipelines, concurrency, and protocols at play.

---

## 📑 Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Service Map & Process Boundaries](#2-service-map--process-boundaries)
3. [The 12-Stage Execution Pipeline](#3-the-12-stage-execution-pipeline)
   - [Stage 1 — Audio Capture (Full-Duplex Stream)](#stage-1--audio-capture-full-duplex-stream)
   - [Stage 2 — Wake Word Detection](#stage-2--wake-word-detection)
   - [Stage 3 — Voice Activity Detection (VAD)](#stage-3--voice-activity-detection-vad)
   - [Stage 4 — STT Timing State Machine](#stage-4--stt-timing-state-machine)
   - [Stage 5 — Speech-to-Text (Whisper TRT)](#stage-5--speech-to-text-whisper-trt)
   - [Stage 6 — The Dual-Protocol Edge Connection](#stage-6--the-dual-protocol-edge-connection)
   - [Stage 7 — FastAPI Orchestrator Server](#stage-7--fastapi-orchestrator-server)
   - [Stage 8 — Hybrid Semantic Router](#stage-8--hybrid-semantic-router)
   - [Stage 9 — Agent Dispatch (A2A Protocol)](#stage-9--agent-dispatch-a2a-protocol)
   - [Stage 10 — Agent Execution Deep Dive](#stage-10--agent-execution-deep-dive)
   - [Stage 11 — LLM Streaming Back Over WebSocket](#stage-11--llm-streaming-back-over-websocket)
   - [Stage 12 — Text-to-Speech (Kokoro ONNX)](#stage-12--text-to-speech-kokoro-onnx)
4. [Frontend WebSocket Logic & 3D Bridge](#4-frontend-websocket-logic--3d-bridge)
5. [Thread & Concurrency Architecture](#5-thread--concurrency-architecture)
6. [Prerequisites & Complete Setup Guide](#6-prerequisites--complete-setup-guide)
7. [Configuration Variables (.env)](#7-configuration-variables-env)
8. [API Reference Reference](#8-api-reference-reference)

---

## 1. High-Level Architecture

The framework consists of two main pillars that communicate via HTTP + WebSockets:

```text
┌────────────────────────────────────────────────────────────────────────────────────┐
│                       NVIDIA JETSON DEVICE / LOCAL HOST                            │
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
│                                            │   ...                               │ │
│                                            └─────────────────────────────────────┘ │
│                                                         │ WS "frontend"             │
│                                            Browser UI ◄─┘                          │
│                                         (React, Vite)                              │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Service Map & Process Boundaries

The system distributes its computational workload across independent services.

| Component | File | Language | Runs As |
|---|---|---|---|
| Voice pipeline entry | `voice_processing/main.py` | Python | Standalone process / Docker |
| Wake word | `voice_processing/wake_word.py` | Python + ONNX | Library |
| VAD | `voice_processing/vad.py` | Python + ONNX | Library |
| STT | `voice_processing/stt.py` | Python + Whisper TRT | Library |
| TTS | `voice_processing/tts.py` | Python + Kokoro ONNX | Library |
| Orchestrator API server | `orchestrator/main.py` | Python FastAPI | Docker service |
| Routing logic | `orchestrator/router.py` | Python | Imported by API |
| WebSocket manager | `orchestrator/ws_manager.py` | Python | Singleton in FastAPI |
| Agent dispatcher | `orchestrator/client.py` | Python (A2A) | Imported by API |
| LLM client | `orchestrator/llm.py` | Python (AsyncOpenAI) | Imported by API |
| Agents (Car, Nav, etc.) | `agents/*/agent_executor.py` | Python | Separate Docker Containers |
| Frontend | `frontend/.../OrchestratorChat.jsx` | React | Browser |

---

## 3. The 12-Stage Execution Pipeline

Every interaction goes through a rigorous, latency-optimized, and parallelized pipeline, spanning exactly 12 execution stages from microphone hardware up to TTS audio emission.

### Stage 1 — Audio Capture (Full-Duplex Stream)
**File:** `voice_processing/main.py`

The system utilizes `sounddevice` to establish a native, full-duplex PCM stream at 48kHz (float32).
```python
with sd.Stream(
    device=(dev_id, dev_id),       # input & output locked to same clock
    samplerate=48000,
    blocksize=2048,
    channels=1,
    callback=audio_callback,
    dtype=np.float32,
):
```
The callback function executes every ~42.6ms, pushing microphone inputs into a locked, fixed-size `CircularBuffer` ring fifo queue safely preventing Out-Of-Memory growth, whilst actively trying to pull generated TTS float32 matrices down to the speaker channel without dropping frames.

### Stage 2 — Wake Word Detection
**File:** `voice_processing/wake_word.py`

Custom ONNX instances evaluate 80ms moving windows running parallel model variants (`hey_dora.onnx`, `hey_doh_ra.onnx`).
The 48kHz stream is resampled in blocks via `librosa/soxr` limits to a flat 16kHz before being fed into `OpenWakeWord`.
**Crucial optimization:** Upon surpassing the `0.4` confidence threshold, alongside playing the confirmation *bloop* sound, it asynchronously fires the Orchestrator WS daemon threads **before the user even begins their request**, effectively hiding handshake latency (100ms+) entirely behind human vocal latency.

### Stage 3 — Voice Activity Detection (VAD)
**File:** `voice_processing/vad.py`

Operating simultaneously alongside STT buffer collation, Silero VAD `.onnx` logic parses every 512-sample (32ms) frame resolving explicit float32 probabilities on human vocal cord detection. It does not manipulate the actual 48kHz raw target audio; it acts purely as the primary chronometer triggering subsequent sequence pipelines.

### Stage 4 — STT Timing State Machine
This stage decides exactly when a spoken query has "finished". The intelligence leverages a dual-tier silence parameter metric avoiding hard stops mid-sentence while keeping response throughput high:
- **`SEGMENT_SILENCE` (0.7s):** When 700ms pass since VAD detected speech, the process slices the audio buffer off and passes the *partial* text to STT early to begin warmups.
- **`TURN_END_SILENCE` (1.5s):** Only invoked if `SEGMENT_SILENCE` successfully initiated. Validates user intent closure, merges final partials, constructs a concrete HTTP POST, and flushes resources. 

### Stage 5 — Speech-to-Text (Whisper TRT)
**File:** `voice_processing/stt.py`
The captured audio segment, cleanly scaled via internal padding structures matching 16kHz constraints, invokes a TensorRT compiled `Whisper-Small` model. Jetson GPU Tensorcores optimize inference allowing massive parallel matrix evaluation pushing throughput up to 10X against PyTorch baseline limits. Custom language tags like `<|en|>` are stripped via regex handlers yielding parsed strings.

### Stage 6 — The Dual-Protocol Edge Connection
**File:** `voice_processing/orchestrator_client.py`
Data ingress and egress purposefully bifurcate across dual transports separating LLM timing dependencies directly off STT closures:
1. **HTTP POST Sync:** Native transmission pushes the compiled Whisper string to FastAPI terminating quickly via `HTTP 202` unlocking localized subroutines.
2. **WebSocket Async Tokenization:** An active daemon receives the dynamic LLM evaluation character-by-character. A specialized token buffer groups characters into sentence chunks (via `.`, `!`, `?` limiters). When a sentence resolves, the local batch is instantaneously fired towards the TTS processor.

*Result:* The local device begins actively speaking the first generated sentence aloud while the second sentence has not even been computed by the LLM yet. 

### Stage 7 — FastAPI Orchestrator Server
**File:** `orchestrator/main.py`
Upon the `HTTP POST /v1/orchestrator/message` initialization, FastAPI invokes `process_request(...)` into a non-blocking `asyncio.create_task()` background worker. The primary POST returns almost immediately, leaving the complex semantic intent mapping resolving seamlessly in isolated loops locking the single-user conversational state contexts dynamically.

### Stage 8 — Hybrid Semantic Router
**File:** `orchestrator/router.py` (The Brains)
Commands like `"Open the trunk and play some rock music"` are evaluated aggressively involving:
1. **Multi-Intent Parsing:** Breaks the compound command via language-specific regex conjunction parsing evaluating `spacy` Part-Of-Speech validators verifying isolated nouns aren't stripped from target verbs.
2. **Translation Layer:** Routes Asian constructs via `Argos` resolving target strings into standardized English forms bridging local sub-agents. 
3. **Hybrid Verification Matrix:** Compares target intentions evaluating *Vector Cosine Similarity embeddings* (`paraphrase-multilingual-MiniLM-L12-v2`) commanding 60% validation weight, overlaid with *BM25 Keyword Sparse matrices* filling the remaining 40%. The highest scores dynamically assign isolated agent vectors.

### Stage 9 — Agent Dispatch (A2A Protocol)
**File:** `orchestrator/client.py`
The Orchestrator dispatches split intents iteratively. Utilizing `asyncio.gather()`, it executes *every evaluated agent requirement completely in parallel* converting intents into A2A JSON-RPC wrappers containing extensive runtime metadata (`session_id`, `favorites`, `translations`). Massive concurrent scale resolves effortlessly over independent docker environments without sequential locks.

### Stage 10 — Agent Execution Deep Dive
Agents resolve user commands with unique behaviors. 

For instance, the **Car Control Agent** operates a two-stage Zero-Shot NLI sequence natively mapping "Open the left door" down into constrained NLI parameter matrices. 
Alternatively, **Reminder / User Preference Agents** operate an **ACK + Background sequence**. To keep user feedback instant, they immediately HTTP POST an ACK ("Processing reminder...") which validates the initial WS stream allowing immediate speech output, then later invoke an isolated background `/v1/orchestrator/agent_callback` inserting concrete success responses post-execution.

### Stage 11 — LLM Streaming Back Over WebSocket
**File:** `orchestrator/api/route_orchestrator.py`
FastAPI generates a full conversational contextual prompt mapping system prompts, user texts, and the newly aggregated raw parallel agent data outcomes. Passed directly into `AsyncOpenAI` interfaces (targeting local `llama.cpp` containers hosting `qwen3` parameters), the framework receives chunked completion deltas yielding single sub-components pushed securely back over the pre-established Websocket to the origin caller limits.

### Stage 12 — Text-to-Speech (Kokoro ONNX)
**File:** `voice_processing/tts.py`
The sentence-boundary flushed buffers resolve into local Kokoro TTS parameters invoking `af_heart` internal embeddings matrices parsing 24kHz audio structures limit. `Resampler` immediately pivots the structure targeting 48kHz blocks mapping linearly against the primary `voice_processing` duplex loop. 2048 block array clusters push physically towards the active audio hardware natively terminating the full system loop.

---

## 4. Frontend WebSocket Logic & 3D Bridge

If a frontend browser is actively connected to the orchestrator (at Websocket endpoint `frontend`), it natively duplicates and intercepts the LLM token bursts generating responsive user interface interactions matching specific Jetson boundaries.

Crucially, **Raw Agent Data results** parsed during Stage 10 broadcast specific payload values directly off to React bounds initializing `CustomEvent('CAR_CONTROL_ACTION')` loops natively turning semantic structures (`open_door_right`) into dynamic local web-based animations visually.

---

## 5. Thread & Concurrency Architecture

The voice processing loop is highly multi-threaded to prevent blocking:

```text
┌─── OS Audio Thread (42.6ms callbacks) ──────────────────────────────┐
│  audio_callback() loops continuous buffer reads & writes            │
└─────────────────────────────────────────────────────────────────────┘
                 ▲ (Thread-safe Queue/Ring buffer)
                 ▼
┌─── process_audio() Thread ──────────────────────────────────────────┐
│  Evaluates WakeWord ONNX -> Triggers VAD -> STT Segmentation       │
└─────────────────────────────────────────────────────────────────────┘
                 ▼ (stt_queue)
┌─── process_stt() Thread ────────────────────────────────────────────┐
│  Executes TensorRT Whisper queries -> Fires HTTP POST (Sync)        │
└─────────────────────────────────────────────────────────────────────┘
┌─── Orchestrator WS Thread (Daemon) ─────────────────────────────────┐
│  Listens to WS, buffers text, forwards to TTS bounds (Async)        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Prerequisites & Complete Setup Guide

- **Python 3.11+**
- **uv** (Python package manager): `pip install uv`
- **Docker and Docker Compose** (highly recommended for deployment)
- **Local LLM Server** (llama-server running GGUF configurations)

### Docker Compose Deployment (Recommended)

1. **Initialization:**
```bash
git clone <repo-url> orchestrator-on-edge
cd orchestrator-on-edge
```

2. **Start Subsystem LLM Resources:** Note `extra_hosts` handles OS differences dynamically. Ensure your .env `LLM_PORT` matches your host configuration parameter (default 8082).
```bash
llama-server -m /path/to/your-model.gguf --port 8082 --host 0.0.0.0
```

3. **Deploy Core Infrastructure:**
```bash
docker compose up --build
```
*The Orchestrator will implement automated reconnect routines waiting for underlying containerized agents to assemble their native environments safely.*

### Native Execution (Development Mode)

Useful for native debugging avoiding Docker abstractions:
1. Adjust `.env` enforcing bounds:
```bash
LOCAL_LLM_URL=http://localhost:8082/v1
AGENT_URLS=http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004,http://localhost:8005
```

2. Synchronize agent environments manually utilizing `uv` inside separate shell contexts:
```bash
# Terminal 1
cd agents/car_control && uv sync && uv run python main.py

# Terminal 2 
cd orchestrator && uv sync && uv run python main.py
```

---

## 7. Configuration Variables (.env)

Adjust key parameters within the `.env` root repository template mapping exact requirements dynamically.

| Setting | Default | Description |
|-----------|-----------|-------------|
| `LLM_PORT` | `8082` | LLM socket port mapped by standard Docker deployments. |
| `LOCAL_LLM_URL` | `http://localhost:8082/v1` | URL for Native implementations matching endpoint bounds mapped manually over host ports. |
| `OPENAI_API_KEY` | (None) | Pre-empts Local bounds querying Cloud OpenAI parameters bypassing local llama instances entirely. |
| `CAR_MANUAL_SEARCH_MODE`| `hybrid` | Evaluator pattern natively governing agent matrices: `bm25`, `embedding`, `hybrid`. |

---

## 8. API Reference Reference

**FastAPI Core Endpoints (Port 8000)**

| Mode | Route | Payload |
|------|-----------|-----------|
| `POST` | `/v1/orchestrator/message` | `{"message": "str", "session_id": "root"}` (Core entry point) |
| `GET` | `/v1/agents` | Retreive dynamic HTTP agent topologies. |
| `WS` | `/v1/orchestrator/ws/{session}`| Daemon token pipeline broadcasting standard string datasets. |

**Agent Intercommunications (Port 8001+, JSON-RPC)**

```bash
# Simulating direct sub-agent message injection evaluation sequences:
curl -X POST http://localhost:8002/ -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {"message": {"parts": [{"kind": "text", "text": "Set AC to 70 degrees"}]}}
}'
```

---
*Created as a comprehensive Deep-Dive replacing summarized definitions for maximum architectural visibility.*
