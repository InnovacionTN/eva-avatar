"""Eva platica: Gemini Live voice conversation driving the Eva avatar.

Mic -> Gemini Live API (audio dialog) -> speakers, while the response audio
envelope drives Eva's lips procedurally (plus idle head sway and blinks) on
the FasterLivePortrait TRT models. Output goes to the OBS virtual camera.

No webcam needed: motion is synthesized, so per-frame cost is only
stitching + warping_spade (~60 ms => ~15 fps on the 30 W RTX 2000 Ada).

Usage (from the FLP package python):
    python eva_platica.py [--preview] [--check] [--voice Aoede] [--model ...]

--check: connect to Gemini, request one spoken reply, save it to out/, exit.
"""
import argparse
import asyncio
import glob
import os
import random
import sys
import threading
import time

import cv2
import numpy as np

# ---- locate the FasterLivePortrait package root ----
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CANDIDATES = glob.glob(os.path.join(PROJECT_ROOT, "third_party", "FLP-win", "**", "configs", "trt_infer.yaml"),
                       recursive=True)
if not CANDIDATES:
    sys.exit("FasterLivePortrait package not found under third_party/FLP-win")
FLP_ROOT = os.path.dirname(os.path.dirname(CANDIDATES[0]))
os.chdir(FLP_ROOT)          # model paths in the yaml are relative to the package root
sys.path.insert(0, FLP_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

SOURCE_IMAGE = os.path.join(PROJECT_ROOT, "assets", "eva_source.png")

SEND_RATE = 16000   # Gemini Live expects 16 kHz PCM16 mono in
RECV_RATE = 24000   # and answers with 24 kHz PCM16 mono

SYSTEM_PROMPT = (
    "Eres Eva, la presentadora virtual de Tiendas Neto, una cadena mexicana de "
    "tiendas de descuento. Hablas siempre en español de México, con un tono "
    "cálido, profesional y cercano. Tus respuestas son breves y conversacionales "
    "(una a tres frases), como en una plática cara a cara. Nunca digas que eres "
    "un modelo de lenguaje; eres Eva."
)

MODEL_CANDIDATES = [
    "gemini-3.1-flash-live-preview",                 # newest, verified 2026-06-11
    "gemini-2.5-flash-native-audio-preview-09-2025",  # previous default, proven
    "gemini-2.5-flash-native-audio-latest",
]


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
class SpeakerStream:
    """Plays Gemini's 24 kHz PCM; exposes loudness envelope + an audio clock
    (played/fed milliseconds) that the phoneme scheduler aligns against."""

    def __init__(self):
        import sounddevice as sd
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._level = 0.0
        self._played = 0  # samples actually sent to the device
        self._fed = 0     # samples received from Gemini
        self.stream = sd.OutputStream(samplerate=RECV_RATE, channels=1, dtype="int16",
                                      blocksize=1200, callback=self._callback)

    def _callback(self, outdata, frames, time_info, status):
        need = frames * 2
        with self._lock:
            chunk = bytes(self._buf[:need])
            del self._buf[:need]
            self._played += len(chunk) // 2
        if len(chunk) < need:
            chunk += b"\x00" * (need - len(chunk))
        samples = np.frombuffer(chunk, dtype=np.int16)
        outdata[:, 0] = samples
        rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        level = min(1.0, rms / 0.12)
        with self._lock:
            # instant attack, fast release: the mouth must close between
            # syllables or speech reads as a sustained open-jaw gape
            self._level = max(level, self._level * 0.72)

    def feed(self, data: bytes):
        with self._lock:
            self._buf.extend(data)
            self._fed += len(data) // 2

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._level = 0.0
            self._fed = self._played  # buffered audio dropped: clocks resync

    def now_ms(self) -> float:
        with self._lock:
            return self._played / RECV_RATE * 1000.0

    def fed_ms(self) -> float:
        with self._lock:
            return self._fed / RECV_RATE * 1000.0

    def speaking(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def level(self) -> float:
        with self._lock:
            return self._level

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


# --------------------------------------------------------------------------
# Spanish phonemes -> viseme schedule (the approach the user picked in the
# A/B/C comparison; see scripts/compare_three.py and deep-research-report.md)
# --------------------------------------------------------------------------
import math
import unicodedata

PHONE_COST = {"a": 1.10, "e": 1.00, "i": 0.95, "o": 1.05, "u": 1.00,
              "m": 0.70, "b": 0.65, "p": 0.65, "f": 0.70,
              "s": 0.65, "x": 0.70, "tS": 0.75, "y": 0.65,
              "t": 0.55, "d": 0.55, "n": 0.55, "l": 0.55, "r": 0.50,
              "k": 0.55, "g": 0.55}
VISEME_OF = {"a": "A", "e": "EI", "i": "EI", "o": "OU", "u": "OU",
             "m": "MBP", "b": "MBP", "p": "MBP", "f": "FV"}
VISEME_NAMES = ["A", "EI", "OU", "MBP", "FV", "C_NEUTRAL"]


def g2p_es(text):
    """Tiny rule-based Spanish grapheme->phoneme; Spanish is near-phonetic."""
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")
    phones = []
    for word in text.split():
        i = 0
        while i < len(word):
            c, nxt = word[i], word[i + 1] if i + 1 < len(word) else ""
            if c == "h":
                pass
            elif c == "c" and nxt == "h":
                phones.append("tS"); i += 1
            elif c == "l" and nxt == "l":
                phones.append("y"); i += 1
            elif c == "r" and nxt == "r":
                phones.append("r"); i += 1
            elif c == "q":
                phones.append("k"); i += 1 + (1 if nxt == "u" else 0)
            elif c == "c":
                phones.append("s" if nxt in "ei" else "k")
            elif c == "g" and nxt in "ei":
                phones.append("x")
            elif c == "j":
                phones.append("x")
            elif c == "z":
                phones.append("s")
            elif c == "v":
                phones.append("b")
            elif c == "y":
                phones.append("i" if not nxt else "y")
            elif c in "aeioubdfgklmnprstx":
                phones.append(c)
            i += 1
    return phones


class PhonemeScheduler:
    """Turns Gemini's output transcription into time-aligned viseme weights.

    Each transcript fragment is scheduled over the audio that is already
    buffered but not yet played (we know exactly how much), so the mouth
    tracks the speech without forced alignment.
    """

    def __init__(self, lead_ms=90.0):
        self._lock = threading.Lock()
        self.events = []          # (start_ms, end_ms, viseme)
        self.last_sched_ms = 0.0
        self.turn_text = ""
        self.prev = {v: 0.0 for v in VISEME_NAMES}
        self.lead_ms = lead_ms    # evaluate ahead of the audio clock

    def on_transcript(self, frag, played_ms, fed_ms):
        if not frag:
            return
        with self._lock:
            # transcription may arrive as fragments or cumulative text
            if self.turn_text and frag.startswith(self.turn_text):
                new = frag[len(self.turn_text):]
                self.turn_text = frag
            else:
                new = frag
                self.turn_text += frag
            phones = g2p_es(new)
            if not phones:
                return
            costs = [PHONE_COST.get(p, 0.60) for p in phones]
            start = max(self.last_sched_ms, played_ms)
            end = max(fed_ms, start + 60.0 * len(phones))
            unit = (end - start) / sum(costs)
            cursor = start
            for p, c in zip(phones, costs):
                self.events.append((cursor, cursor + c * unit, VISEME_OF.get(p, "C_NEUTRAL")))
                cursor += c * unit
            self.last_sched_ms = cursor
            # drop events long past
            self.events = [e for e in self.events if e[1] > played_ms - 2000.0]

    def clear(self, played_ms):
        with self._lock:
            self.events = []
            self.last_sched_ms = played_ms
            self.turn_text = ""

    def weights(self, t_ms):
        t_ms = t_ms + self.lead_ms
        with self._lock:
            events = list(self.events)
            prev = self.prev
        raw = {v: 0.0 for v in VISEME_NAMES}
        for a, b, v in events:
            start = a - (40.0 if v in ("A", "EI", "OU") else 15.0)  # anticipation
            if t_ms < start or t_ms > b:
                continue
            # narrow window: influence dies off between phones so the mouth
            # visibly relaxes between syllables instead of holding open
            center, half = 0.5 * (start + b), max(15.0, 0.40 * (b - start))
            w = math.exp(-0.5 * ((t_ms - center) / half) ** 2)
            raw[v] = max(raw[v], w)
        # a bilabial closure must beat neighboring vowels, never blend away
        if raw["MBP"] > 0.25:
            damp = 1.0 - 0.8 * raw["MBP"]
            for v in VISEME_NAMES:
                if v != "MBP":
                    raw[v] *= damp
        s = sum(raw.values())
        if s > 1.0:
            raw = {k: v / s for k, v in raw.items()}
        out = {}
        for v in VISEME_NAMES:
            alpha = 0.70 if raw[v] > prev[v] else 0.38  # attack / release
            step = np.clip(alpha * (raw[v] - prev[v]), -0.18, 0.18)
            out[v] = prev[v] + float(step)
        with self._lock:
            self.prev = out
        return out


class MicStream:
    """Captures 16 kHz PCM16 mono and hands chunks to the asyncio sender."""

    def __init__(self, loop, queue, gate=None, device=None):
        import sounddevice as sd
        self.loop = loop
        self.queue = queue
        self.gate = gate  # callable -> True when mic should be muted
        self.stream = sd.InputStream(samplerate=SEND_RATE, channels=1, dtype="int16",
                                     blocksize=1600, device=device, callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        if self.gate is not None and self.gate():
            return
        data = bytes(indata.tobytes())
        self.loop.call_soon_threadsafe(self.queue.put_nowait, data)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


# --------------------------------------------------------------------------
# Gemini Live session (runs on its own asyncio thread)
# --------------------------------------------------------------------------
class GeminiVoice(threading.Thread):
    def __init__(self, speaker, model=None, voice=None, allow_barge_in=False,
                 mic_device=None, scheduler=None, vision=False, vision_camera=0,
                 vision_fps=1.0):
        super().__init__(daemon=True)
        self.speaker = speaker
        self.model = model
        self.voice = voice
        self.allow_barge_in = allow_barge_in
        self.mic_device = mic_device
        self.scheduler = scheduler
        self.vision = vision
        self.vision_camera = vision_camera
        self.vision_fps = vision_fps
        self.connected = threading.Event()
        self.fatal = None

    def _client(self):
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (expected in .env)")
        # verified 2026-06-11: this key works on the Developer API endpoint,
        # NOT Vertex express (its project lacks the Vertex service)
        return genai.Client(api_key=key)

    def _config(self):
        from google.genai import types
        speech = None
        if self.voice:
            speech = types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)))
        prompt = SYSTEM_PROMPT
        if self.vision:
            prompt += (" Recibes imágenes en vivo de la cámara del equipo: puedes ver "
                       "lo que pasa frente a la computadora. Si te preguntan qué ves, "
                       "describe la imagen más reciente con naturalidad.")
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=prompt)]),
            speech_config=speech,
            # text of what Eva says, as she says it -> drives the phoneme mouth
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:  # surface the reason in the render loop
            self.fatal = e
            self.connected.set()

    async def _main(self):
        from google.genai import types
        client = self._client()
        config = self._config()
        candidates = [self.model] if self.model else MODEL_CANDIDATES
        last_err = None
        for model in candidates:
            try:
                async with client.aio.live.connect(model=model, config=config) as session:
                    print(f"[gemini] connected: {model}")
                    self.connected.set()
                    await self._session_loop(session, types)
                    return
            except Exception as e:
                last_err = e
                print(f"[gemini] {model}: {type(e).__name__}: {e}")
        raise RuntimeError(f"could not connect to any Live model: {last_err}")

    async def _session_loop(self, session, types):
        mic_q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        # without echo cancellation the speakers feed back into the mic; by
        # default mute the mic while Eva talks (no barge-in on open speakers)
        gate = None if self.allow_barge_in else self.speaker.speaking
        mic = MicStream(loop, mic_q, gate=gate, device=self.mic_device)
        mic.start()
        print("[gemini] mic live - habla con Eva")

        async def sender():
            while True:
                chunk = await mic_q.get()
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SEND_RATE}"))

        async def receiver():
            while True:
                async for message in session.receive():
                    sc = getattr(message, "server_content", None)
                    if sc is not None and getattr(sc, "interrupted", False):
                        self.speaker.clear()
                        if self.scheduler is not None:
                            self.scheduler.clear(self.speaker.now_ms())
                    if message.data:
                        self.speaker.feed(message.data)
                    if sc is not None and self.scheduler is not None:
                        tr = getattr(sc, "output_transcription", None)
                        if tr is not None and getattr(tr, "text", None):
                            self.scheduler.on_transcript(tr.text, self.speaker.now_ms(),
                                                         self.speaker.fed_ms())

        async def vision_sender():
            cap = cv2.VideoCapture(self.vision_camera, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                print("[vision] camera not available - continuing without sight")
                return
            print("[vision] camera streaming to Gemini (~1 fps)")
            try:
                while True:
                    ok, frame = cap.read()
                    if ok:
                        h, w = frame.shape[:2]
                        if w > 768:
                            frame = cv2.resize(frame, (768, int(h * 768 / w)))
                        ok2, jpg = cv2.imencode(".jpg", frame,
                                                [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok2:
                            await session.send_realtime_input(
                                video=types.Blob(data=jpg.tobytes(), mime_type="image/jpeg"))
                    await asyncio.sleep(1.0 / self.vision_fps)
            finally:
                cap.release()

        tasks = [sender(), receiver()]
        if self.vision:
            tasks.append(vision_sender())
        try:
            await asyncio.gather(*tasks)
        finally:
            mic.stop()


# --------------------------------------------------------------------------
# avatar
# --------------------------------------------------------------------------
class OutputSink(threading.Thread):
    """Compose + virtual-cam send + preview on their own thread so the GPU
    render loop never waits on them."""

    def __init__(self, cam, preview, canvas_base, side, x_off, y_off):
        super().__init__(daemon=True)
        self.cam = cam
        self.preview = preview
        self.canvas_base = canvas_base
        self.side, self.x_off, self.y_off = side, x_off, y_off
        self.cond = threading.Condition()
        self.face = None
        self.running = True
        self.quit_requested = False

    def submit(self, face):
        with self.cond:
            self.face = face
            self.cond.notify()

    def run(self):
        while self.running:
            with self.cond:
                while self.face is None and self.running:
                    self.cond.wait(0.1)
                face, self.face = self.face, None
            if face is None:
                continue
            canvas = self.canvas_base.copy()
            canvas[self.y_off:self.y_off + self.side, self.x_off:self.x_off + self.side] = \
                cv2.resize(face, (self.side, self.side))
            if self.cam is not None:
                self.cam.send(canvas)
            if self.preview:
                cv2.imshow("Eva platica", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.quit_requested = True
        if self.preview:
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify()


class EvaRenderer:
    def __init__(self, source, pasteback=True, lip_gain=0.8, sway=1.0, smile_gain=0.0):
        from omegaconf import OmegaConf
        from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

        cfg = OmegaConf.load(os.path.join(FLP_ROOT, "configs", "trt_infer.yaml"))
        cfg.infer_params.flag_pasteback = pasteback
        print("[init] loading TensorRT engines...")
        self.pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
        if not self.pipe.prepare_source(source, realtime=True):
            sys.exit(f"no face detected in source: {source}")

        (self.x_s_info, self.source_lmk, self.R_s, self.f_s, self.x_s, self.x_c_s,
         _lip_delta0, _flag_lip0, self.mask_ori, self.M) = self.pipe.src_infos[0][0]
        self.pasteback = pasteback and self.mask_ori is not None
        self.lip_gain = lip_gain
        self.sway = sway
        self.smile_gain = smile_gain

        import torch
        self.torch = torch
        self.src_rgb = self.pipe.src_imgs[0]
        self.base_tensor = torch.from_numpy(self.src_rgb).to(self.pipe.device).float()

        # source lip/eye openness; animation moves relative to these
        from src.utils.utils import calc_lip_close_ratio, calc_eye_close_ratio
        self.c_s_lip = float(calc_lip_close_ratio(self.source_lmk[None])[0][0])
        self.c_s_eye = float(calc_eye_close_ratio(self.source_lmk[None])[0][:2].mean())

        # mined from the example driving videos by scripts/extract_smile_delta.py;
        # off by default (user verdict: creepy), enable with --smile-gain > 0
        self.smile_delta = None
        smile_path = os.path.join(PROJECT_ROOT, "assets", "smile_delta.npy")
        if smile_gain > 0 and os.path.exists(smile_path):
            self.smile_delta = np.load(smile_path).astype(self.x_s.dtype)
            print("[init] smile delta loaded")

        # real speech mouth shapes (scripts/extract_visemes.py): jaw + interior
        # move like actual talking; retarget_lip alone reads as clenched teeth
        self.visemes = None
        self.templates = None
        vis_path = os.path.join(PROJECT_ROOT, "assets", "visemes.npy")
        if os.path.exists(vis_path):
            self.visemes = np.load(vis_path).astype(self.x_s.dtype)
            ah, ee, oh = self.visemes[0], self.visemes[1], self.visemes[2]
            # viseme classes as mixes of the mined real shapes (won the A/B/C test);
            # consonants stay nearly closed so the mouth shuts between syllables
            self.templates = {"A": ah * 0.85, "EI": ee * 0.75, "OU": oh * 0.95,
                              "MBP": ah * -0.35, "FV": ah * 0.08, "C_NEUTRAL": ah * 0.10}
            print(f"[init] {len(self.visemes)} visemes loaded")
        self.cur_vis = 0
        self.prev_level = 0.0
        self.smile_s = 0.0
        self.last_t = time.time()

        self.t0 = time.time()
        self.next_blink = self.t0 + random.uniform(1.5, 3.0)
        self.blink_t = None
        warm = self.frame(0.0)  # build engines' first-run kernels before going live
        print(f"[init] avatar ready ({warm.shape[1]}x{warm.shape[0]})")

    def envelope_delta(self, lip_level):
        """Fallback mouth driver: loudness envelope switches mined shapes."""
        if self.visemes is None:
            return None
        if self.prev_level < 0.30 <= lip_level:
            self.cur_vis = random.choices(range(len(self.visemes)),
                                          weights=[0.5, 0.25, 0.25][:len(self.visemes)])[0]
        self.prev_level = lip_level
        return self.visemes[self.cur_vis] * ((lip_level ** 1.2) * self.lip_gain)

    def weights_delta(self, weights):
        """Phoneme-scheduler mouth driver: blend templates by viseme weights."""
        delta = np.zeros_like(self.x_s_info["exp"])
        for name, w in weights.items():
            if w > 0.001 and name in self.templates:
                delta = delta + self.templates[name] * (w * self.lip_gain)
        return delta

    def _blink_ratio(self, tnow, speaking):
        # long enough that the closed eyes land on rendered frames at ~10 fps
        BLINK_DUR = 0.22
        if self.blink_t is None and tnow >= self.next_blink:
            self.blink_t = tnow
        if self.blink_t is None:
            return None
        p = (tnow - self.blink_t) / BLINK_DUR
        if p >= 1.0:
            self.blink_t = None
            gap = random.uniform(3.0, 6.0) if speaking else random.uniform(2.0, 4.2)
            self.next_blink = tnow + gap
            return None
        return self.c_s_eye - (self.c_s_eye - 0.03) * float(np.sin(np.pi * p))

    def frame(self, lip_level, speaking=False, mouth_delta=None):
        """Render one RGB frame. mouth_delta (exp delta) wins over lip_level."""
        from src.utils.utils import transform_keypoint
        from src.utils.crop import paste_back_pytorch
        s = self.x_s_info
        tnow = time.time() - self.t0

        dyaw = self.sway * (2.0 * np.sin(2 * np.pi * tnow / 8.7) + 0.7 * np.sin(2 * np.pi * tnow / 3.1))
        dpitch = self.sway * 1.1 * np.sin(2 * np.pi * tnow / 6.3)
        droll = self.sway * 0.6 * np.sin(2 * np.pi * tnow / 11.2)

        # smile eases in when she's quiet, eases mostly out while she talks
        now = time.time()
        dt = min(0.2, now - self.last_t)
        self.last_t = now
        smile_target = 0.15 if speaking else 1.0
        self.smile_s += (smile_target - self.smile_s) * (1.0 - np.exp(-dt / 0.6))

        exp = s["exp"]
        if self.smile_delta is not None:
            exp = exp + self.smile_delta * (self.smile_s * self.smile_gain)

        if mouth_delta is None and self.visemes is not None:
            mouth_delta = self.envelope_delta(lip_level)
        if mouth_delta is not None:
            exp = exp + mouth_delta

        x_d = transform_keypoint(s["pitch"] + dpitch, s["yaw"] + dyaw, s["roll"] + droll,
                                 s["t"], exp, s["scale"], s["kp"])

        if mouth_delta is None:
            # fallback: lip-openness retargeting (less natural: no jaw motion)
            c_d_lip = self.c_s_lip + (lip_level ** 1.4) * self.lip_gain
            comb_lip = self.pipe.calc_combined_lip_ratio([c_d_lip], self.source_lmk)
            lip_delta = self.pipe.retarget_lip(self.x_s, comb_lip)
            x_d = x_d + lip_delta.reshape(-1, self.x_s.shape[1], 3)

        blink = self._blink_ratio(time.time(), speaking)
        if blink is not None:
            comb_eye = self.pipe.calc_combined_eye_ratio([[blink]], self.source_lmk)
            eye_delta = self.pipe.retarget_eye(self.x_s, comb_eye)
            x_d = x_d + eye_delta.reshape(-1, self.x_s.shape[1], 3)

        x_d = self.pipe.stitching(self.x_s, x_d)
        out = self.pipe.model_dict["warping_spade"].predict(self.f_s, self.x_s, x_d)
        if self.pasteback:
            out = paste_back_pytorch(out, self.M, self.base_tensor.clone(), self.mask_ori)
        return out.to(dtype=self.torch.uint8).cpu().numpy()


# --------------------------------------------------------------------------
def run_check(args):
    """Headless connectivity test: one text prompt -> spoken reply saved to out/."""
    speaker_buf = bytearray()

    class FakeSpeaker:
        def feed(self, data):
            speaker_buf.extend(data)

        def clear(self):
            pass

        def speaking(self):
            return False

    async def check():
        from google.genai import types
        gv = GeminiVoice(FakeSpeaker(), model=args.model, voice=args.voice)
        client = gv._client()
        config = gv._config()
        candidates = [args.model] if args.model else MODEL_CANDIDATES
        for model in candidates:
            try:
                async with client.aio.live.connect(model=model, config=config) as session:
                    print(f"[check] connected: {model}")
                    await session.send_client_content(turns=types.Content(
                        role="user", parts=[types.Part(text="Preséntate en una frase.")]))
                    async for message in session.receive():
                        if message.data:
                            speaker_buf.extend(message.data)
                        sc = getattr(message, "server_content", None)
                        if sc is not None and getattr(sc, "turn_complete", False):
                            break
                    return model
            except Exception as e:
                print(f"[check] {model}: {type(e).__name__}: {e}")
        return None

    model = asyncio.run(check())
    if not model or not speaker_buf:
        sys.exit("[check] FAILED: no audio received")
    out_dir = os.path.join(PROJECT_ROOT, "out")
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "gemini_check.wav")
    import wave
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RECV_RATE)
        w.writeframes(bytes(speaker_buf))
    secs = len(speaker_buf) / 2 / RECV_RATE
    print(f"[check] OK: {model} returned {secs:.1f}s of audio -> {wav_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_IMAGE)
    ap.add_argument("--model", default=None)
    ap.add_argument("--voice", default=None, help="e.g. Aoede, Kore, Puck, Leda")
    ap.add_argument("--mic-device", default=None, help="sounddevice input index/name")
    ap.add_argument("--vision", action="store_true",
                    help="stream the webcam to Gemini so Eva can see")
    ap.add_argument("--vision-camera", type=int, default=0)
    ap.add_argument("--vision-fps", type=float, default=1.0)
    ap.add_argument("--allow-barge-in", action="store_true",
                    help="keep the mic open while Eva speaks (use headphones!)")
    ap.add_argument("--out-width", type=int, default=1280)
    ap.add_argument("--out-height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--lip-gain", type=float, default=1.25,
                    help="mouth amplitude multiplier")
    ap.add_argument("--mouth", choices=["fonemas", "envelope"], default="fonemas",
                    help="mouth driver: phoneme scheduler (default) or loudness envelope")
    ap.add_argument("--lip-lead", type=float, default=120.0,
                    help="ms the mouth runs ahead of the audio (raise if lips feel late)")
    ap.add_argument("--smile-gain", type=float, default=0.0)
    ap.add_argument("--sway", type=float, default=1.0, help="idle motion amount (0 = statue)")
    ap.add_argument("--no-pasteback", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-virtual-cam", action="store_true")
    ap.add_argument("--check", action="store_true", help="connectivity test only, no avatar")
    ap.add_argument("--mute", action="store_true", help="render avatar without Gemini (lip test)")
    ap.add_argument("--max-frames", type=int, default=0, help="exit after N frames (testing)")
    args = ap.parse_args()

    if args.check:
        run_check(args)
        return

    renderer = EvaRenderer(args.source, pasteback=not args.no_pasteback,
                           lip_gain=args.lip_gain, sway=args.sway,
                           smile_gain=args.smile_gain)

    speaker = None
    scheduler = None
    if not args.mute:
        speaker = SpeakerStream()
        speaker.start()
        if args.mouth == "fonemas" and renderer.templates is not None:
            scheduler = PhonemeScheduler(lead_ms=args.lip_lead)
        mic_dev = None
        if args.mic_device is not None:
            mic_dev = int(args.mic_device) if args.mic_device.isdigit() else args.mic_device
        voice = GeminiVoice(speaker, model=args.model, voice=args.voice,
                            allow_barge_in=args.allow_barge_in, mic_device=mic_dev,
                            scheduler=scheduler, vision=args.vision,
                            vision_camera=args.vision_camera, vision_fps=args.vision_fps)
        voice.start()
        voice.connected.wait(timeout=60)
        if voice.fatal:
            sys.exit(f"[gemini] failed: {voice.fatal}")
        print(f"[init] mouth driver: {'fonemas' if scheduler else 'envelope'}")

    # ---- output canvas ----
    OW, OH = args.out_width, args.out_height
    side = min(OW, OH)
    x_off, y_off = (OW - side) // 2, (OH - side) // 2
    bg = cv2.GaussianBlur(cv2.resize(renderer.src_rgb, (OW, OH)), (0, 0), 25)
    canvas_base = bg.copy()

    cam = None
    if not args.no_virtual_cam:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=OW, height=OH, fps=args.fps,
                                  fmt=pyvirtualcam.PixelFormat.RGB)
        print(f"[init] virtual camera: {cam.device} {OW}x{OH}@{args.fps}")

    sink = OutputSink(cam, args.preview, canvas_base, side, x_off, y_off)
    sink.start()

    frame_budget = 1.0 / args.fps
    n_done, fps_now = 0, 0.0
    fps_t0, fps_n0 = time.time(), 0
    demo_level = 0.0

    print("[run] Eva platica. Ctrl+C to stop.")
    try:
        while True:
            t_start = time.time()
            mouth_delta = None
            if args.mute:  # synthetic syllables to eyeball the lip mapping
                demo_level = max(0.0, demo_level * 0.7 + (random.random() < 0.35) * random.uniform(0.4, 1.0))
                level = min(1.0, demo_level)
                speaking = level > 0.1
            else:
                level = speaker.level()
                speaking = speaker.speaking()
                if scheduler is not None:
                    mouth_delta = renderer.weights_delta(scheduler.weights(speaker.now_ms()))
            face = renderer.frame(level, speaking, mouth_delta=mouth_delta)
            sink.submit(face)
            if sink.quit_requested:
                break

            n_done += 1
            if args.max_frames and n_done >= args.max_frames:
                break
            if n_done - fps_n0 >= 60:
                tnow = time.time()
                fps_now = (n_done - fps_n0) / (tnow - fps_t0)
                fps_t0, fps_n0 = tnow, n_done
                print(f"[run] {fps_now:.1f} fps | frames {n_done}")
            if not args.mute and voice.fatal:
                sys.exit(f"[gemini] session died: {voice.fatal}")
            leftover = frame_budget - (time.time() - t_start)
            if leftover > 0:
                time.sleep(leftover)
    except KeyboardInterrupt:
        pass
    finally:
        sink.stop()
        sink.join(timeout=2)
        if cam is not None:
            cam.close()
        if speaker is not None:
            speaker.stop()
        print(f"[done] {n_done} frames")


if __name__ == "__main__":
    main()
