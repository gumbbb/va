import json
import os

import requests

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


def _parse_ws_url(base_url):
    base = (base_url or "").strip().rstrip("/")
    if base.startswith("https://"):
        return base.replace("https://", "wss://", 1)
    if base.startswith("http://"):
        return base.replace("http://", "ws://", 1)
    return "ws://" + base if base else ""


import threading
import queue

class OrchestratorConnection:
    def __init__(self, base_url=None, session_id=None, on_segment=None):
        self.base_url = (base_url or os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")).rstrip("/")
        self.session_id = session_id or f"voice-{threading.get_ident()}-{int(time.time())}"
        self.on_segment = on_segment
        self.ws = None
        self.ws_thread = None
        self.buffer = []
        self.SENTENCE_END = (".", "!", "?")
        self.is_done = threading.Event()
        self.had_tokens = False
        self._done_count = 0

    def connect(self):
        """Pre-establish WebSocket connection."""
        if not HAS_WEBSOCKET:
            print("[orchestrator_client] websocket-client not installed")
            return
        
        ws_url = _parse_ws_url(self.base_url) + f"/v1/orchestrator/ws/{self.session_id}"
        print(f"[orchestrator_client] Connecting to WS: {ws_url}")

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
                
                if not self.had_tokens:
                    done_msg = d.get("message", "").strip()
                    if done_msg and self.on_segment:
                        try:
                            self.on_segment(done_msg)
                        except Exception as e:
                            print(f"[orchestrator_client] on_segment error: {e}")
                
                self.had_tokens = False
                
                # We signal is_done so that logic like send_and_stream can proceed,
                # but the socket itself remains open for agent callbacks.
                self.is_done.set()

        def on_error(ws, err):
            print(f"[orchestrator_client] WS Error: {err}")

        def on_close(ws, status, msg):
            print("[orchestrator_client] WS Closed")
            self.flush()
            self.is_done.set()

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval": 20}, daemon=True)
        self.ws_thread.start()

    def flush(self):
        if not self.buffer: return
        segment = "".join(self.buffer).strip()
        self.buffer.clear()
        if segment and self.on_segment:
            try:
                self.on_segment(segment)
            except Exception as e:
                print(f"[orchestrator_client] on_segment error: {e}")

    def send_message(self, user_message, post_timeout=30):
        """Send the actual message via POST."""
        post_url = f"{self.base_url}/v1/orchestrator/message"
        try:
            r = requests.post(
                post_url,
                json={"message": user_message, "session_id": self.session_id},
                timeout=post_timeout,
                headers={"Content-Type": "application/json"}
            )
            r.raise_for_status()
            print(f"[orchestrator_client] Sent OK (HTTP {r.status_code})")
            return True
        except Exception as e:
            print(f"[orchestrator_client] POST failed: {e}")
            return False

    def close(self):
        if self.ws:
            try: self.ws.close()
            except: pass
        self.is_done.set()

def send_and_stream(user_message, session_id, on_segment, **kwargs):
    """Legacy compatibility wrapper."""
    conn = OrchestratorConnection(session_id=session_id, on_segment=on_segment)
    conn.connect()
    # Give a small buffer for WS to establish if needed
    import time
    time.sleep(0.2)
    conn.send_message(user_message)
    # Wait for completion or timeout
    conn.is_done.wait(timeout=kwargs.get("ws_timeout", 60))
    conn.close()
