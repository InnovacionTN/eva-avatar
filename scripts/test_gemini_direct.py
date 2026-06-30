"""Minimal, direct Gemini Live multi-turn test — no web layer at all.
Replicates how eva_platica uses the session: connect, stream audio via
send_realtime_input, read responses. Feeds two TTS utterances as two turns.

If this prints 2 turns, the bug is in eva_web's structure; if 1, it's the
fundamental send pattern / config.
"""
import asyncio
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import eva_platica as ep
from google.genai import types


def load16k(path):
    w = wave.open(path, "rb")
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if w.getframerate() != 16000:
        idx = (np.arange(int(len(a) * 16000 / w.getframerate())) * w.getframerate() / 16000).astype(int)
        a = a[idx]
    return a.astype(np.int16).tobytes()


SPEECH1 = load16k(os.path.join(ep.PROJECT_ROOT, "out", "utt1.wav"))
SPEECH2 = load16k(os.path.join(ep.PROJECT_ROOT, "out", "utt2.wav"))


async def main():
    gv = ep.GeminiVoice(None, vision=False)
    client = gv._client()
    config = gv._config()
    config.realtime_input_config = types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
            prefix_padding_ms=200, silence_duration_ms=600))
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.1-flash-live-preview"
    async with client.aio.live.connect(model=model, config=config) as session:
        print("[direct] connected", model)
        st = {"turns": 0, "bytes": 0}

        async def receiver():
            async for m in session.receive():
                if m.data:
                    st["bytes"] += len(m.data)
                sc = getattr(m, "server_content", None)
                if sc is not None and getattr(sc, "turn_complete", False):
                    st["turns"] += 1
                    print(f"[direct] <- turn_complete #{st['turns']} ({st['bytes']//1024} kB)")

        turn_done = asyncio.Event()

        async def receiver2():
            while True:                          # re-enter receive() each turn
                async for m in session.receive():
                    if m.data:
                        st["bytes"] += len(m.data)
                    sc = getattr(m, "server_content", None)
                    if sc is not None and getattr(sc, "turn_complete", False):
                        st["turns"] += 1
                        print(f"[direct] <- turn_complete #{st['turns']} ({st['bytes']//1024} kB)")
                        turn_done.set()

        rt = asyncio.create_task(receiver2())
        import time
        z = bytes(3200)

        async def stream(pcm):
            # 0.3s lead sil + speech + 1.5s trailing sil (lets VAD endpoint)
            data = z * 3 + pcm + z * 15
            chunks = [data[i:i + 3200] for i in range(0, len(data), 3200)]
            T0 = time.monotonic()
            for n, c in enumerate(chunks):
                if len(c) < 3200:
                    c = c + bytes(3200 - len(c))
                await session.send_realtime_input(
                    audio=types.Blob(data=c, mime_type="audio/pcm;rate=16000"))
                dt = T0 + (n + 1) * 0.1 - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)

        async def do_turn(pcm, label):
            print(f"[direct] -> {label}; then SILENT while she replies")
            turn_done.clear()
            await stream(pcm)
            try:
                await asyncio.wait_for(turn_done.wait(), timeout=15)
            except asyncio.TimeoutError:
                print(f"[direct] !! no reply to {label} within 15s")

        await do_turn(SPEECH1, "TURN 1")
        await asyncio.sleep(0.8)
        await do_turn(SPEECH2, "TURN 2")
        await asyncio.sleep(1)
        rt.cancel()
        print(f"[direct] RESULT: {st['turns']} turn(s) -> "
              f"{'PASS' if st['turns'] >= 2 else 'FAIL'}")


asyncio.run(main())
