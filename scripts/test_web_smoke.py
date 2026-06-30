"""Headless smoke test for eva_web: connect a WS client, send a bit of silence,
confirm Eva video frames stream back. No browser needed.

Assumes the server is already running on --port (default 8080).
"""
import asyncio
import sys

import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
TAG_AUDIO, TAG_VIDEO, TAG_FLUSH = 0, 1, 2


async def main():
    uri = f"ws://localhost:{PORT}/ws"
    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
        print(f"[smoke] connected to {uri}")
        # send 0.5 s of 16k silence as a mic frame
        silence = bytes(1 + 16000)  # tag byte + 8000 int16 zeros
        sil = bytearray(silence)
        sil[0] = TAG_AUDIO
        await ws.send(bytes(sil))

        video, audio = 0, 0
        try:
            while video < 10:
                msg = await asyncio.wait_for(ws.recv(), timeout=12)
                if isinstance(msg, str):
                    print("[smoke] text:", msg)
                    continue
                tag = msg[0]
                if tag == TAG_VIDEO:
                    video += 1
                    if video == 1:
                        print(f"[smoke] first Eva video frame: {len(msg) - 1} bytes JPEG")
                elif tag == TAG_AUDIO:
                    audio += 1
        except (asyncio.TimeoutError, TimeoutError):
            pass
        print(f"[smoke] received {video} video frames, {audio} audio chunks")
        print("[smoke] RESULT:", "PASS" if video >= 5 else "FAIL (no video stream)")


asyncio.run(main())
