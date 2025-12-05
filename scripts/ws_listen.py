"""Simple WebSocket listener that prints received frames.

Usage: python scripts/ws_listen.py <session_id>
"""
import asyncio
import json
import sys
import websockets

async def listen(ws_url):
    async with websockets.connect(ws_url) as ws:
        print("Connected:", ws_url)
        try:
            while True:
                msg = await ws.recv()
                try:
                    data = json.loads(msg)
                except Exception:
                    data = msg
                print("MSG:", data)
        except Exception as e:
            print("Websocket closed:", e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ws_listen.py <session_id>")
        raise SystemExit(1)
    sid = sys.argv[1]
    url = f"ws://localhost:8000/streams/{sid}"
    asyncio.run(listen(url))
