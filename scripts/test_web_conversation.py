"""Drive the eva_web WebSocket like a browser: stream two real speech turns
(separated by silence) and report how many times Eva replies. Isolates the
web pipeline's multi-turn behavior from any phone/browser/echo factor.

Server must be running on --port (default 8080).
"""
import asyncio
import sys
import time
import wave

import numpy as np
import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
TAG_AUDIO, TAG_VIDEO, TAG_FLUSH = 0, 1, 2

def load16k(path):
    w = wave.open(path, "rb")
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if w.getframerate() != 16000:
        idx = (np.arange(int(len(a) * 16000 / w.getframerate())) * w.getframerate() / 16000).astype(int)
        a = a[idx]
    return a.astype(np.int16).tobytes()

# two DIFFERENT human-ish utterances (Windows TTS), one per turn
SPEECH1 = load16k("out/utt1.wav")
SPEECH2 = load16k("out/utt2.wav")
SIL_100 = bytes(3200)            # 100 ms of 16 kHz silence
print(f"[conv] turn1 {len(SPEECH1)/2/16000:.1f}s  turn2 {len(SPEECH2)/2/16000:.1f}s")


async def main():
    uri = f"ws://localhost:{PORT}/ws"
    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
        print(f"[conv] connected {uri}")
        rx = {"bytes": 0, "bursts": 0, "last": 0.0}

        async def receiver():
            while True:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)) and msg and msg[0] == TAG_AUDIO:
                    now = time.time()
                    if now - rx["last"] > 1.2:       # a >1.2s gap = a new reply
                        rx["bursts"] += 1
                        print(f"[conv] <- Eva reply #{rx['bursts']} starts (t={now-T0:.1f}s)")
                    rx["last"] = now
                    rx["bytes"] += len(msg) - 1

        async def send_pcm(pcm, label):
            print(f"[conv] -> {label} ({len(pcm)/2/16000:.1f}s) t={time.time()-T0:.1f}s")
            for i in range(0, len(pcm), 3200):
                await ws.send(bytes([TAG_AUDIO]) + pcm[i:i + 3200])
                await asyncio.sleep(0.1)             # realtime pacing

        async def send_silence(secs):
            for _ in range(int(secs * 10)):
                await ws.send(bytes([TAG_AUDIO]) + SIL_100)
                await asyncio.sleep(0.1)

        T0 = time.time()
        rxt = asyncio.create_task(receiver())
        await send_silence(0.5)
        await send_pcm(SPEECH1, "TURN 1 speech")
        await send_silence(10)        # let Eva reply (~6s) + echo-guard window
        await send_pcm(SPEECH2, "TURN 2 speech")
        await send_silence(10)
        rxt.cancel()
        print(f"[conv] RESULT: Eva replied {rx['bursts']} time(s), {rx['bytes']//1024} kB audio")
        print("[conv]", "PASS (multi-turn works)" if rx["bursts"] >= 2 else "FAIL (only one reply)")


asyncio.run(main())
