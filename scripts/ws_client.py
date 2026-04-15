"""
Connect to orchestrator WebSocket and print streaming messages.
Usage: python ws_client.py [session_id]
Default session_id: test-1
Then in another terminal: curl -X POST http://localhost:8000/v1/orchestrator/message -H "Content-Type: application/json" -d '{"message": "open the door", "session_id": "test-1"}'
"""
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Install: pip install websockets (or uv add websockets)")
    sys.exit(1)


async def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else "test-1"
    url = f"ws://localhost:8000/v1/orchestrator/ws/{session_id}"
    print(f"Connecting to {url} ... (send message with same session_id to see stream)")
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            d = json.loads(msg)
            t = d.get("type", "")
            if t == "token":
                print(d.get("content", ""), end="", flush=True)
            elif t == "done":
                full = d.get("message", "")
                print("\n[done]", full[:200] + ("..." if len(full) > 200 else ""))
            elif t == "error":
                print("\n[error]", d.get("message", ""))
            else:
                print("\n", d)


if __name__ == "__main__":
    asyncio.run(main())
