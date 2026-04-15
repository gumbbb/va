# Edge Orchestrator — Complete Architecture Deep Dive

## 1. System Overview

Two independent services communicate over HTTP + WebSocket:

| Service | Location | Role |
|---|---|---|
| **voice_processing** | `voice_processing/` | Mic capture, Wake Word, VAD, STT, TTS — runs on Jetson |
| **orchestrator** | `orchestrator-on-edge/orchestrator/` | FastAPI server: routing, agents, LLM streaming |

Nine AI agents run as separate microservices under `orchestrator-on-edge/agents/`:
`car_control`, `navigation`, `infotainment`, `car_manual`, `cloud`, `user_preference`, `reminder`, `telegram`, `smart_home`

---

## 2. Complete Data Flow Diagram

```
┌─────────────────── VOICE CLIENT (voice_processing/) ───────────────────┐
│                                                                         │
│  Microphone (48kHz, int16)                                              │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────┐    audio_callback() converts float32→int16             │
│  │ sd.Stream   │──▶ input_buffer (CircularBuffer, ring FIFO)            │
│  │ (duplex)    │◀── output_queue (TTS playback chunks)                  │
│  └─────────────┘                                                        │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────────── Thread: process_audio() ────────────────────┐    │
│  │                                                                  │    │
│  │  1. WakeWordProcessor.update_state(chunk)                        │    │
│  │     ├─ Resample 48kHz → 16kHz                                    │    │
│  │     ├─ OpenWakeWord ONNX (hey_dora/hey_doh_ra/hey_dola)          │    │
│  │     ├─ Threshold: 0.4, debounce: 0.5s                           │    │
│  │     └─ On detect: play "listening" sound                         │    │
│  │                    PRE-CONNECT WebSocket to orchestrator ←──┐    │    │
│  │                    (OrchestratorConnection.connect())       │    │    │
│  │                                                             │    │    │
│  │  2. VADProcessor.update_buffer(chunk) + get_prob()          │    │    │
│  │     ├─ Silero VAD ONNX, 512-sample chunks at 16kHz          │    │    │
│  │     └─ prob > 0.5 → voice detected                          │    │    │
│  │                                                             │    │    │
│  │  3. Timing State Machine:                                   │    │    │
│  │     ├─ SEGMENT_SILENCE (0.7s) → push audio to stt_queue     │    │    │
│  │     ├─ TURN_END_SILENCE (1.5s) → push END_TURN sentinel     │    │    │
│  │     └─ VAD_FIRST_TIMEOUT (3.0s) → play timeout, reset       │    │    │
│  └──────────────────────────────────────────────────────────────┘    │
│       │                                                                 │
│       ▼ stt_queue                                                       │
│  ┌──────────────────── Thread: process_stt() ──────────────────────┐    │
│  │                                                                  │    │
│  │  For each audio segment:                                         │    │
│  │    STTProcessor.transcribe(audio)                                │    │
│  │    ├─ Resample 48kHz → 16kHz                                     │    │
│  │    ├─ Whisper TRT (TensorRT-optimized)                           │    │
│  │    ├─ Auto-detect language (vi/en/ja/ko)                         │    │
│  │    └─ Accumulate partial texts in turn_texts[]                   │    │
│  │                                                                  │    │
│  │  On END_TURN:                                                    │    │
│  │    text = " ".join(turn_texts)                                   │    │
│  │    active_conn.send_message(text)  ◄── HTTP POST                 │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌──── OrchestratorConnection (WS thread) ────┐                         │
│  │  on_message:                                │                         │
│  │    "token" → buffer.append(content)         │                         │
│  │              if ends with .!? → flush()     │                         │
│  │    "done"  → flush() + is_done.set()  ←─────┤── THE BUG IS HERE      │
│  │    "error" → flush() + is_done.set()        │                         │
│  │                                             │                         │
│  │  flush() → on_segment(sentence) → TTS       │                         │
│  └─────────────────────────────────────────────┘                         │
│                                                                         │
│  TTSProcessor.speech(segment)                                           │
│    ├─ Kokoro ONNX (24kHz output)                                        │
│    ├─ Resample 24kHz → 48kHz                                            │
│    └─ Chunk into 2048-sample frames → output_queue                      │
└─────────────────────────────────────────────────────────────────────────┘

                          │ HTTP POST                    ▲ WebSocket
                          ▼                              │

┌─────────────────── ORCHESTRATOR (FastAPI) ──────────────────────────────┐
│                                                                         │
│  POST /v1/orchestrator/message                                          │
│    ├─ OrchestratorRouter.route(user_message)                            │
│    │   ├─ MultiIntentParser.parse() — split compound commands           │
│    │   ├─ SimpleArgosTranslator — translate to English                  │
│    │   ├─ SentenceTransformer.encode() — embed query                    │
│    │   └─ hybrid_scores = 0.6*cosine + 0.4*BM25 — rank agents          │
│    │                                                                    │
│    ├─ Return {"accepted": true, "agent_ids": [...]} immediately         │
│    └─ asyncio.create_task(process_request(...))  ← background           │
│                                                                         │
│  process_request():                                                     │
│    ├─ send_to_agents_parallel()                                         │
│    │   ├─ build_agent_batches() from routing_result.parts               │
│    │   ├─ For each agent: A2A SendMessageRequest via httpx              │
│    │   └─ asyncio.gather(*coros) — all agents in parallel               │
│    │                                                                    │
│    ├─ post_process_agent_response()                                     │
│    │   ├─ Collect {agent_id, success, message} per agent                │
│    │   └─ Aggregate data: urls, images, actions                         │
│    │                                                                    │
│    ├─ WS send: {"type": "data", content: aggregated_response}           │
│    │   └─ This triggers 3D car model via CAR_CONTROL_ACTION event       │
│    │                                                                    │
│    └─ stream_llm(session_id, messages)                                  │
│        ├─ asyncio.Lock per session (prevents concurrent streams)        │
│        ├─ LLM prompt = system + user_request + agent_responses          │
│        ├─ For each token: WS send {"type": "token", content}            │
│        └─ Final: WS send {"type": "done", message: full_text}           │
│                                                                         │
│  ┌─── Agent Callback Path (ASYNC, DELAYED) ───┐                        │
│  │  POST /v1/orchestrator/agent_callback       │                        │
│  │    ├─ Acquires same session Lock             │                        │
│  │    │   (waits for stream_llm to finish)      │                        │
│  │    └─ queue_manager sends:                   │                        │
│  │       {"type": "done", message: callback_msg}│ ← SECOND "done"       │
│  └─────────────────────────────────────────────┘                        │
│                                                                         │
│  _send_to_session_and_frontend():                                       │
│    ├─ ws_manager.send(session_id, msg)  → voice client WS               │
│    └─ if "voice-*": ws_manager.send("frontend", msg) → browser WS      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Agent Categories by Response Pattern

### Category A: Synchronous (single response)
These agents return ONE response via A2A `event_queue`, then the orchestrator does LLM + done.

| Agent | How it works |
|---|---|
| **car_control** | BM25 filter → Zero-shot NLI → execute action → one response |
| **navigation** | LLM intent parse → POI/directions lookup → one response |
| **infotainment** | Zero-shot classify → joke/music lookup → one response |
| **car_manual** | RAG search → one response |
| **cloud** | Web search → one response |

**WS message sequence for these:**
```
data  → agent results (action data, URLs, images)
token → LLM token 1
token → LLM token 2
...
done  → LLM complete (full_text)     ← ONLY ONE "done"
```

### Category B: ACK + Callback (TWO responses)
These agents return an immediate ACK, then fire a SECOND response via `POST /agent_callback` after async processing.

| Agent | ACK message | Callback message |
|---|---|---|
| **reminder** | "Reminder request received and processing." | "Added reminder 'X' successfully" |
| **user_preference** | "Preference request received and processing." | "Saved X to favorite Y" |
| **smart_home** | "Smart home request received and processing" | LLM analysis of camera image |
| **telegram** | "Telegram request received and processing" | Send result |

**WS message sequence for these:**
```
data  → agent ACK response (the placeholder message)
token → LLM token 1 (based on placeholder ACK)
token → LLM token 2
...
done  → LLM complete                  ← FIRST "done"
--- session lock released ---
done  → agent callback real result     ← SECOND "done"
```

---

## 4. THE BUG: Why TTS Only Speaks the First Response

### Root Cause

In `orchestrator_client.py` line 57-59:

```python
elif t in ("done", "error"):
    self.flush()
    self.is_done.set()   # ← KILLS THE CONNECTION
```

When the **first** `"done"` arrives (the LLM stream finishing), the voice client:
1. Flushes remaining buffered tokens to TTS ✓
2. Sets `is_done` event ✓
3. **The WS thread effectively stops processing** ✗

The **second** `"done"` (the agent callback with the REAL result) arrives AFTER `is_done` is set. At this point:

- In `main.py`, there is no code that waits for or handles a second `"done"`.
- The `process_stt()` thread has already moved on after `send_message()` returned.
- The `active_conn` may have already been replaced by the next wake word.

### Timeline of the Bug

```
T+0.0s  Wake word → WS connect (session voice-abc123)
T+2.0s  STT done → POST /message "remind me to buy milk"
T+2.1s  Orchestrator routes → reminder agent
T+2.2s  Reminder ACK: {"success": true, "message": "Reminder request received..."}
T+2.3s  Orchestrator sends data + starts LLM stream
T+2.5s  LLM tokens arrive → TTS speaks: "Sure, I'm processing your reminder."
T+3.0s  LLM done → WS: {"type": "done"}
         → voice client: flush() + is_done.set()  ← CONNECTION CONSIDERED DEAD
T+3.5s  Reminder background completes: "Added reminder 'buy milk' successfully"
T+3.6s  POST /agent_callback → orchestrator
T+3.7s  Session lock released → WS: {"type": "done", "message": "Added reminder..."}
         → voice client: is_done already set, WS thread still running but...
         → flush() IS called, on_segment(segment) IS called
         → BUT: the segment is the raw callback message, NOT an LLM-summarized response
```

**Wait — actually the WS thread IS still running** (it's `run_forever` in a daemon thread). So the second `done` DOES get processed, `flush()` IS called, and `on_segment` IS called. Let me re-examine...

The actual issue is more nuanced. The WS `run_forever` loop continues. The `on_message` handler processes the second `done`:

```python
elif t in ("done", "error"):
    self.flush()        # ← flushes empty buffer (no tokens preceded this "done")
    self.is_done.set()  # ← already set, no effect
```

**The buffer is EMPTY** because the second `"done"` from `queue_manager` sends:
```python
{"type": "done", "message": "Added reminder 'buy milk' successfully"}
```

But the `on_message` handler only buffers `"token"` type messages. The `"done"` message's `message` field is NEVER read or passed to TTS. The handler only calls `self.flush()` which flushes `self.buffer` — which is empty because no `"token"` events preceded this second `"done"`.

### The Exact Bug (Precise)

**`on_message` in `orchestrator_client.py` ignores the `message` field of `"done"` events.**

```python
if t == "token":
    content = d.get("content", "")    # ← reads "content"
    self.buffer.append(content)       # ← buffers it
    ...
elif t in ("done", "error"):
    self.flush()                      # ← flushes buffer (EMPTY for callbacks)
    self.is_done.set()                # ← d.get("message") is NEVER used
```

The callback `"done"` arrives with `{"type": "done", "message": "Added reminder..."}` but the voice client never extracts `d["message"]` — it just flushes an empty buffer and ignores the actual text.

---

## 5. How the Frontend Handles It Correctly

The frontend (`OrchestratorChat.jsx` lines 249-288) handles multiple `"done"` messages properly:

```javascript
if (data.type === "done") {
    setChatHistory(prev => {
        const existingMsg = prev.find(m => m.id === resolvedId)

        // Case 1: still streaming → finalize (LLM final chunk)
        if (existingMsg && existingMsg.streaming) {
            return prev.map(m => m.id === resolvedId
                ? { ...m, streaming: false, content: data.message || m.content }
                : m
            )
        }

        // Case 2: NOT streaming → this is a NEW message (agent callback)
        const newMessage = {
            id: uuid(),           // ← NEW bubble
            content: data.message // ← reads the message field!
        }
        return [...prev, newMessage]
    })
}
```

The frontend creates a **new chat bubble** for each subsequent `"done"`. The voice client throws it away.

---

## 6. The Fix

### Option A: Read `message` field from `done` events (Minimal Fix)

In `voice_processing/orchestrator_client.py`, modify `on_message`:

```python
def on_message(ws, raw):
    try:
        d = json.loads(raw)
    except: return
    t = d.get("type", "")
    if t == "token":
        content = d.get("content", "")
        if content:
            self.buffer.append(content)
            text = "".join(self.buffer).rstrip()
            if text.endswith(self.SENTENCE_END):
                self.flush()
    elif t in ("done", "error"):
        self.flush()  # flush any remaining token buffer

        # ★ NEW: speak the "message" field if present and buffer was empty
        done_msg = d.get("message", "").strip()
        if done_msg and self.on_segment:
            # Only speak if this is a callback (not the LLM final text,
            # which was already streamed via tokens)
            # Heuristic: if buffer was empty before flush, this is a callback
            try:
                self.on_segment(done_msg)
            except Exception as e:
                print(f"[orchestrator_client] on_segment error: {e}")

        self.is_done.set()
```

**Problem:** This would DOUBLE-speak the LLM response because the LLM `"done"` also has `message: full_text` AND the tokens were already spoken via buffer flush.

### Option B: Track whether tokens were streamed (Recommended Fix)

```python
class OrchestratorConnection:
    def __init__(self, ...):
        ...
        self.had_tokens = False  # ★ NEW

    def on_message(ws, raw):
        ...
        if t == "token":
            self.had_tokens = True  # ★ mark that we got tokens
            content = d.get("content", "")
            if content:
                self.buffer.append(content)
                text = "".join(self.buffer).rstrip()
                if text.endswith(self.SENTENCE_END):
                    self.flush()
        elif t in ("done", "error"):
            self.flush()

            # ★ If no tokens preceded this "done", it's a callback — speak it
            done_msg = d.get("message", "").strip()
            if not self.had_tokens and done_msg and self.on_segment:
                try:
                    self.on_segment(done_msg)
                except Exception as e:
                    print(f"[orchestrator_client] on_segment error: {e}")

            self.had_tokens = False  # ★ reset for next "done"
            # DON'T set is_done on first done — wait for potential callbacks
```

**But this still has a problem:** How do we know if MORE callbacks are coming?

### Option C: Don't close on first done, use timeout (Production Fix)

```python
class OrchestratorConnection:
    def __init__(self, ...):
        ...
        self.had_tokens = False
        self._done_count = 0

    def connect(self):
        ...
        def on_message(ws, raw):
            try:
                d = json.loads(raw)
            except: return
            t = d.get("type", "")
            if t == "token":
                self.had_tokens = True
                content = d.get("content", "")
                if content:
                    self.buffer.append(content)
                    text = "".join(self.buffer).rstrip()
                    if text.endswith(self.SENTENCE_END):
                        self.flush()
            elif t in ("done", "error"):
                self.flush()
                self._done_count += 1

                # Speak callback messages (done events without preceding tokens)
                if not self.had_tokens:
                    done_msg = d.get("message", "").strip()
                    if done_msg and self.on_segment:
                        try:
                            self.on_segment(done_msg)
                        except Exception as e:
                            print(f"[orchestrator_client] on_segment error: {e}")

                self.had_tokens = False  # reset for next batch
                # Don't set is_done immediately — let timeout handle it
        ...
```

Then in `main.py`, instead of relying on `is_done`, use a timeout after the WS goes idle.

---

## 7. Which Agents Are Affected

| Agent | Uses Callback? | TTS speaks real result? |
|---|---|---|
| car_control | No | ✅ Yes (LLM summarizes action) |
| navigation | No | ✅ Yes |
| infotainment | No | ✅ Yes |
| car_manual | No | ✅ Yes |
| cloud | No | ✅ Yes |
| **reminder** | **Yes** | ❌ Only speaks "processing..." |
| **user_preference** | **Yes** | ❌ Only speaks "processing..." |
| **smart_home** | **Yes** | ❌ Only speaks "processing..." |
| **telegram** | **Yes** | ❌ Only speaks "processing..." |
