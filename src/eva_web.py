"""Eva web: browser-based conversational avatar over WebSocket.

This PC is the GPU server; the user's *browser* is the mic, camera, speakers
and screen. We reuse the local renderer + Gemini lip-sync unchanged and only
replace the I/O layer (sounddevice / OpenCV / pyvirtualcam) with a WebSocket:

    browser mic   (16 kHz PCM16)  --ws-->  Gemini Live (audio in)
    browser cam   (JPEG ~1 fps)   --ws-->  Gemini Live (vision)
    Eva voice     (24 kHz PCM16)  <-ws--   Gemini reply (also drives lips)
    Eva face      (JPEG @ fps)     <-ws--   EvaRenderer on the GPU

One conversation at a time (single GPU). Run with the FLP package python:
    python eva_web.py [--port 8080] [--source assets/eva_body.png]
Then open the printed URL (or a tunnel URL) in a browser.
"""
import argparse
import asyncio
import json
import os
import threading
import time

import cv2
import numpy as np

# importing eva_platica performs the FLP setup (chdir into the package, sys.path,
# load .env) and gives us the renderer + Gemini + lip-sync logic to reuse
import eva_platica as ep
from eva_platica import (PhonemeScheduler, SEND_RATE, RECV_RATE, PROJECT_ROOT)

from aiohttp import web, WSMsgType

# binary message tags (first byte of every binary frame)
TAG_AUDIO = 0   # browser->server: mic PCM16@16k ; server->browser: Eva PCM16@24k
TAG_VIDEO = 1   # browser->server: camera JPEG    ; server->browser: Eva JPEG
TAG_FLUSH = 2   # server->browser: drop buffered audio (Eva was interrupted)

WEB_DIR = os.path.join(PROJECT_ROOT, "web")


class WebSpeaker:
    """Holds Gemini's 24 kHz reply for lip-sync timing (loudness envelope + a
    play clock) while the *browser* does the actual playback. The clock is
    advanced in real time by the render loop's tick()."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buf = bytearray()   # not-yet-"played" samples, for the envelope
        self._level = 0.0
        self._played = 0          # samples the clock has advanced past
        self._fed = 0             # samples received from Gemini
        self._t_last = None
        self._last_active = 0.0    # last time there was audio to play

    def feed(self, data: bytes):
        with self._lock:
            self._buf.extend(data)
            self._fed += len(data) // 2

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._level = 0.0
            self._fed = self._played

    def tick(self):
        now = time.time()
        with self._lock:
            if self._t_last is None:
                self._t_last = now
                return
            dt = now - self._t_last
            self._t_last = now
            take = min(int(dt * RECV_RATE) * 2, len(self._buf))
            if take > 0:
                chunk = bytes(self._buf[:take])
                del self._buf[:take]
                self._played += take // 2
                s = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(s ** 2))) if s.size else 0.0
                self._level = max(min(1.0, rms / 0.12), self._level * 0.72)
            else:
                self._level *= 0.72
            if len(self._buf) > 0:
                self._last_active = now

    def busy(self, hangover=1.0):
        # True while Eva's audio is (or was just) playing — covers the browser's
        # playback lag so her echo can't open a user turn
        with self._lock:
            return len(self._buf) > 0 or (time.time() - self._last_active) < hangover

    def now_ms(self):
        with self._lock:
            return self._played / RECV_RATE * 1000.0

    def fed_ms(self):
        with self._lock:
            return self._fed / RECV_RATE * 1000.0

    def speaking(self):
        with self._lock:
            return len(self._buf) > 0

    def level(self):
        with self._lock:
            return self._level


async def ws_handler(request):
    app = request.app
    renderer = app["renderer"]
    args = app["args"]
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)

    if app["busy"]:
        await ws.send_json({"type": "error", "msg": "Eva is busy with another session."})
        await ws.close()
        return ws
    app["busy"] = True
    print("[web] client connected")

    loop = asyncio.get_running_loop()
    speaker = WebSpeaker()
    scheduler = PhonemeScheduler(lead_ms=args.lip_lead) if renderer.templates is not None else None
    meter = ep.UserVoiceMeter()
    mic_q = asyncio.Queue()
    video_q = asyncio.Queue(maxsize=2)
    out_q = asyncio.Queue()
    state = {"running": True}
    st = {"gem_audio": 0, "turns": 0}

    # reuse GeminiVoice only for its client/config helpers
    vision_on = app["vision"]
    gv = ep.GeminiVoice(None, model=args.model, voice=args.voice, vision=vision_on)
    from google.genai import types
    client = gv._client()
    config = gv._config()   # automatic activity detection (Gemini's VAD), as local
    candidates = [args.model] if args.model else ep.MODEL_CANDIDATES

    async def browser_recv():
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                tag, payload = msg.data[0], msg.data[1:]
                if tag == TAG_AUDIO:
                    s = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
                    if s.size:
                        meter.update(min(1.0, float(np.sqrt(np.mean(s ** 2))) / 0.08))
                    # mute the mic while Eva talks AND for ~1.5s after, so her
                    # echo off the phone speaker never reaches Gemini
                    if not speaker.busy(1.5):
                        mic_q.put_nowait(payload)
                elif tag == TAG_VIDEO:
                    if video_q.full():
                        try:
                            video_q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    video_q.put_nowait(payload)
            elif msg.type == WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                except ValueError:
                    continue
                if cmd.get("gesture"):
                    renderer.trigger_gesture(cmd["gesture"])
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
        state["running"] = False

    async def browser_send():
        while state["running"]:
            tag, data = await out_q.get()
            await ws.send_bytes(bytes([tag]) + data)

    async def gemini_session():
        last_err = None
        for model in candidates:
            try:
                async with client.aio.live.connect(model=model, config=config) as session:
                    print(f"[web] gemini connected: {model}")

                    async def sender():
                        # batch the browser's tiny fragments to ~100 ms (the
                        # cadence the local app uses) and stream continuously;
                        # Gemini's automatic VAD finds the turns
                        buf = bytearray()
                        CHUNK = SEND_RATE // 10 * 2   # 100 ms of 16-bit PCM
                        while state["running"]:
                            buf.extend(await mic_q.get())
                            if len(buf) >= CHUNK:
                                data = bytes(buf); buf.clear()
                                await session.send_realtime_input(
                                    audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_RATE}"))

                    async def vision():
                        while state["running"]:
                            jpg = await video_q.get()
                            await session.send_realtime_input(
                                video=types.Blob(data=jpg, mime_type="image/jpeg"))

                    async def receiver():
                        # session.receive() yields one turn then ends; re-enter
                        # it each turn or only the first reply ever arrives
                        while state["running"]:
                            async for message in session.receive():
                                sc = getattr(message, "server_content", None)
                                if sc is not None and getattr(sc, "interrupted", False):
                                    speaker.clear()
                                    if scheduler is not None:
                                        scheduler.clear(speaker.now_ms())
                                    out_q.put_nowait((TAG_FLUSH, b""))
                                if message.data:
                                    st["gem_audio"] += len(message.data)
                                    speaker.feed(message.data)
                                    out_q.put_nowait((TAG_AUDIO, message.data))
                                if sc is not None and getattr(sc, "turn_complete", False):
                                    st["turns"] += 1
                                    print(f"[web] <- turn {st['turns']} complete "
                                          f"({st['gem_audio']//1024} kB)", flush=True)
                                if sc is not None and scheduler is not None:
                                    tr = getattr(sc, "output_transcription", None)
                                    if tr is not None and getattr(tr, "text", None):
                                        scheduler.on_transcript(tr.text, speaker.now_ms(),
                                                                speaker.fed_ms())

                    async def guarded(name, coro):
                        try:
                            await coro
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"[web] task {name} DIED: {type(e).__name__}: {e}", flush=True)
                            raise

                    tasks = [guarded("sender", sender()), guarded("receiver", receiver())]
                    if vision_on:
                        tasks.append(guarded("vision", vision()))
                    await asyncio.gather(*tasks)
                    return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_err = e
                print(f"[web] {model}: {type(e).__name__}: {e}")
        print(f"[web] gemini failed: {last_err}")
        state["running"] = False

    def render_thread():
        budget = 1.0 / args.fps
        th = args.height
        while state["running"]:
            t0 = time.time()
            speaker.tick()
            level = speaker.level()
            speaking = speaker.speaking()
            listening = meter.speaking() and not speaking
            mouth = None
            if scheduler is not None:
                mouth = renderer.weights_delta(scheduler.weights(speaker.now_ms()))
            face = renderer.frame(level, speaking, mouth_delta=mouth, listening=listening)
            h, w = face.shape[:2]
            small = cv2.resize(face, (int(round(w * th / h)), th), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", small[:, :, ::-1],   # RGB->BGR for correct colors
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if ok and out_q.qsize() < 8:                        # drop frames if browser lags
                loop.call_soon_threadsafe(out_q.put_nowait, (TAG_VIDEO, jpg.tobytes()))
            leftover = budget - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)

    rt = threading.Thread(target=render_thread, daemon=True)
    rt.start()
    recv_task = asyncio.create_task(browser_recv())
    send_task = asyncio.create_task(browser_send())
    gem_task = asyncio.create_task(gemini_session())
    try:
        await recv_task                      # returns when the browser disconnects
    finally:
        state["running"] = False
        for tk in (send_task, gem_task):
            tk.cancel()
        await asyncio.gather(send_task, gem_task, return_exceptions=True)
        rt.join(timeout=2)
        app["busy"] = False
        print("[web] client disconnected", flush=True)
    return ws


async def index_handler(request):
    # no-cache so kiosk/Android browsers never serve a stale build
    return web.FileResponse(
        os.path.join(WEB_DIR, "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                 "Pragma": "no-cache", "Expires": "0"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(PROJECT_ROOT, "assets", "eva_body.png"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default=None)
    ap.add_argument("--voice", default="Leda")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--height", type=int, default=720, help="streamed video height (px)")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--lip-gain", type=float, default=1.25)
    ap.add_argument("--lip-lead", type=float, default=120.0)
    ap.add_argument("--sway", type=float, default=1.0)
    ap.add_argument("--body-sway", type=float, default=1.6)
    ap.add_argument("--body-breath", type=float, default=1.0)
    ap.add_argument("--no-vision", action="store_true",
                    help="disable sending the camera to Gemini (audio-only)")
    args = ap.parse_args()

    if not os.path.isabs(args.source):
        args.source = os.path.join(PROJECT_ROOT, args.source)
    body_motion = "body" in os.path.basename(args.source).lower()

    renderer = ep.EvaRenderer(args.source, pasteback=True, lip_gain=args.lip_gain,
                              sway=args.sway, body_motion=body_motion,
                              body_sway=args.body_sway, body_breath=args.body_breath)

    app = web.Application()
    app["renderer"] = renderer
    app["args"] = args
    app["busy"] = False
    app["vision"] = not args.no_vision
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/web", WEB_DIR)

    print(f"[web] Eva server on http://localhost:{args.port}  (open in a browser)")
    print("[web] for a remote tester, start a tunnel to this port (cloudflared/ngrok)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
