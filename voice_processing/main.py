#!/usr/bin/env python3
"""
Standalone pipeline: Wake word -> VAD -> STT -> Orchestrator (stream) -> TTS.
Orchestrator tokens are buffered every n chunks (default 5), each segment is sent to TTS.
"""
import os
import sys
import queue
import time
import uuid
import numpy as np
import sounddevice as sd
import threading
from threading import Thread, Event

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_utils import CircularBuffer, get_valid_input_device_id
from notification_sounds import play_into_queue as play_notification
from wake_word import WakeWordProcessor
from vad import VADProcessor
from stt import STTProcessor
from tts import TTSProcessor
from orchestrator_client import send_and_stream, OrchestratorConnection

DEVICE_SAMPLE_RATE = 48000
CHANNELS = 1
CHUNK = 2048
VAD_FIRST_TIMEOUT = 3.0
SEGMENT_SILENCE = float(os.environ.get("SEGMENT_SILENCE", "0.7"))
TURN_END_SILENCE = 1.5
MIN_SEGMENT_SEC = float(os.environ.get("MIN_SEGMENT_SEC", "0.095"))
WAKE_WORD_DEBOUNCE = 0.5
BUFFER_MAX = 300 * CHUNK
TTS_CHUNK_COUNT = int(os.environ.get("TTS_CHUNK_COUNT", "5"))


def run():
    dev_id, dev_info = get_valid_input_device_id(os.getenv("MIC_DEVICE_NAME"))
    if dev_id is None:
        dev_id = sd.default.device[0]
        dev_info = sd.query_devices(dev_id)
    device_name = dev_info.get("name", "?") if dev_info else "?"

    ww = WakeWordProcessor(DEVICE_SAMPLE_RATE)
    vad = VADProcessor(DEVICE_SAMPLE_RATE)
    stt = STTProcessor(DEVICE_SAMPLE_RATE, model_name="small", language=None)
    tts = TTSProcessor(DEVICE_SAMPLE_RATE)

    input_buffer = CircularBuffer(maxsize=BUFFER_MAX)
    output_queue = queue.Queue()
    stt_queue = queue.Queue(maxsize=5)
    stop = Event()

    END_TURN = object()

    ww_detected = False
    vad_detected_once = False
    timeout_at = None
    last_voice_time = None
    pending_turn = False
    collected_speech = np.array([], dtype=np.int16)
    last_wake_word_time = 0.0
    active_conn = None
    active_conn_lock = threading.Lock()

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            pass
        chunk = (indata[:, 0] * 32768).astype(np.int16)
        input_buffer.put(chunk)
        if not output_queue.empty():
            outdata[:] = output_queue.get_nowait()
        else:
            outdata.fill(0)

    def process_audio():
        nonlocal ww_detected, vad_detected_once, timeout_at, last_voice_time, pending_turn, collected_speech, last_wake_word_time, active_conn
        while not stop.is_set():
            try:
                chunk = input_buffer.get(CHUNK)
                now = time.time()
                if chunk is not None:
                    ww.update_state(chunk.tobytes())
                    if ww.get_detected() and (now - last_wake_word_time) >= WAKE_WORD_DEBOUNCE:
                        last_wake_word_time = now
                        ww_detected = True
                        vad_detected_once = False
                        last_voice_time = None
                        pending_turn = False
                        collected_speech = np.array([], dtype=np.int16)
                        vad.empty_buffer()
                        play_notification("listening", output_queue, CHUNK, DEVICE_SAMPLE_RATE)

                        # Swap in the new connection under lock, then close the old one
                        # asynchronously so ws.close() never blocks process_audio.
                        with active_conn_lock:
                            old_conn = active_conn

                            session_id = f"voice-{uuid.uuid4().hex[:12]}"

                            def on_segment(segment):
                                tts.speech(
                                    segment,
                                    language=os.getenv("LANG", "en"),
                                    output_queue=output_queue,
                                    chunk_size=CHUNK,
                                    device_sample_rate=DEVICE_SAMPLE_RATE,
                                )

                            active_conn = OrchestratorConnection(session_id=session_id, on_segment=on_segment)
                            active_conn.connect()

                        # Close old connection outside the lock and in a background thread
                        # so ws.close() cannot block process_audio or expire timeout_at.
                        if old_conn:
                            threading.Thread(target=old_conn.close, daemon=True).start()

                        # Set timeout AFTER setup so the full VAD_FIRST_TIMEOUT window
                        # is always available regardless of how long connect() took.
                        timeout_at = time.time() + VAD_FIRST_TIMEOUT
                    if ww_detected:
                        vad.update_buffer(chunk.tobytes())
                        collected_speech = np.concatenate((collected_speech, chunk))

                if ww_detected:
                    p = vad.get_prob()
                    if p > 0.5:
                        if not vad_detected_once:
                            print(f"[VAD] Voice detected! (p={p:.2f})")
                        vad_detected_once = True
                        last_voice_time = now

                if last_voice_time is not None:
                    segment_at = last_voice_time + SEGMENT_SILENCE
                    turn_end_at = last_voice_time + TURN_END_SILENCE
                    if pending_turn and now >= turn_end_at:
                        try:
                            stt_queue.put(END_TURN, block=False)
                        except queue.Full:
                            print("[STT] ERROR: stt_queue is FULL (END_TURN dropped)")
                        play_notification("sending", output_queue, CHUNK, DEVICE_SAMPLE_RATE)
                        ww_detected = False
                        timeout_at = None
                        last_voice_time = None
                        pending_turn = False
                        collected_speech = np.array([], dtype=np.int16)
                    elif now >= segment_at and len(collected_speech) > 0:
                        dur = len(collected_speech) / DEVICE_SAMPLE_RATE
                        if dur >= MIN_SEGMENT_SEC:
                            try:
                                stt_queue.put(collected_speech.copy(), block=False)
                            except queue.Full:
                                print(f"[STT] ERROR: stt_queue is FULL (dropped {dur:.2f}s segment)")
                            pending_turn = True
                        collected_speech = np.array([], dtype=np.int16)
                    elif now >= turn_end_at and not pending_turn:
                        print(f"[VAD] Idle timeout after voice (now={now:.2f}, end={turn_end_at:.2f})")
                        play_notification("timeout", output_queue, CHUNK, DEVICE_SAMPLE_RATE)
                        ww_detected = False
                        timeout_at = None
                        last_voice_time = None
                        collected_speech = np.array([], dtype=np.int16)
                elif timeout_at is not None and now >= timeout_at:
                    print(f"[VAD] NO VOICE timeout (now={now:.2f}, timeout={timeout_at:.2f})")
                    play_notification("timeout", output_queue, CHUNK, DEVICE_SAMPLE_RATE)
                    ww_detected = False
                    timeout_at = None
                    collected_speech = np.array([], dtype=np.int16)

                if chunk is None:
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    def process_stt():
        nonlocal active_conn
        allowed_langs = ["vi", "en", "ja", "ko"]
        turn_texts = []
        turn_lang = None
        while not stop.is_set():
            try:
                item = stt_queue.get(timeout=1.0)
                if item is END_TURN:
                    text = " ".join(turn_texts).strip()
                    turn_texts = []
                    if not text:
                        turn_lang = None
                        continue
                    lang = turn_lang
                    print(f"[STT] Full turn: '{text}' (lang={lang})")
                else:
                    audio = item
                    dur = len(audio) / DEVICE_SAMPLE_RATE
                    t0 = time.time()
                    out = stt.transcribe(audio)
                    part = out.get("text", "").strip()
                    if part:
                        turn_texts.append(part)
                    if out.get("language"):
                        l = out.get("language")
                        if l in allowed_langs:
                            turn_lang = l
                        else:
                            # If not allowed, default to None (auto) or keep previous
                            pass
                    print(f"[STT] Segment done in {time.time()-t0:.2f}s: '{part}' (lang={out.get('language')})")
                    continue
                if not text:
                    continue
                if os.environ.get("ORCHESTRATOR_URL"):
                    conn_to_use = None
                    with active_conn_lock:
                        conn_to_use = active_conn

                    if conn_to_use:
                        if not text:
                            with active_conn_lock:
                                if active_conn == conn_to_use:
                                    active_conn.close()
                                    active_conn = None
                            continue
                        
                        print(f"[main] Sending turn to orchestrator: '{text}' (session={conn_to_use.session_id})")
                        t0_send = time.time()
                        success = conn_to_use.send_message(text)
                        print(f"[main] Message sent in {time.time()-t0_send:.2f}s (success={success})")
                        if not success:
                            print("[main] Failed to send message to orchestrator")
                    else:
                        # Fallback if somehow connection wasn't made
                        session_id = f"voice-{uuid.uuid4().hex[:12]}"
                        send_and_stream(
                                text,
                                session_id,
                                lambda s: tts.speech(s, language=lang, output_queue=output_queue, chunk_size=CHUNK, device_sample_rate=DEVICE_SAMPLE_RATE),
                                n_chunks=TTS_CHUNK_COUNT,
                            )
                else:
                    tts.speech(
                        text,
                        language=lang,
                        output_queue=output_queue,
                        chunk_size=CHUNK,
                        device_sample_rate=DEVICE_SAMPLE_RATE,
                    )
            except queue.Empty:
                continue
            except Exception:
                pass

    t_audio = Thread(target=process_audio, daemon=True)
    t_stt = Thread(target=process_stt, daemon=True)
    t_audio.start()
    t_stt.start()

    try:
        with sd.Stream(
            device=(dev_id, dev_id),
            samplerate=DEVICE_SAMPLE_RATE,
            blocksize=CHUNK,
            channels=CHANNELS,
            callback=audio_callback,
            dtype=np.float32,
        ):
            while not stop.is_set():
                sd.sleep(100)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        t_audio.join(timeout=2)
        t_stt.join(timeout=2)



if __name__ == "__main__":
    run()
