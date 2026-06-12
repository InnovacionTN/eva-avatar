"""Render the same phrase with three mouth-driving approaches, same renderer.

A_envelope      - production approach: loudness envelope -> mined viseme bank
B_fonemas       - research-report approach: Spanish phonemes scheduled on the audio clock
C_joyvasa_labios- JoyVASA generated motion, lip keypoints ONLY (no machine head/eyes)

Everything else (blinks, head sway, renderer, audio) is identical across the
three, so the user judges purely the mouth. Outputs in out/compare/.
"""
import glob
import math
import os
import subprocess
import sys
import unicodedata
import wave

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CAND = glob.glob(os.path.join(PROJECT_ROOT, "third_party", "FLP-win", "**", "configs", "trt_infer.yaml"),
                 recursive=True)
FLP_ROOT = os.path.dirname(os.path.dirname(CAND[0]))
os.chdir(FLP_ROOT)
sys.path.insert(0, FLP_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import transform_keypoint, calc_lip_close_ratio, calc_eye_close_ratio
from src.utils.crop import paste_back_pytorch

WAV = os.path.join(PROJECT_ROOT, "out", "compare", "phrase.wav")
PHRASE = "hola soy eva el murcielago rojo corre velozmente"
SOURCE = os.path.join(PROJECT_ROOT, "assets", "eva_source.png")
OUT_DIR = os.path.join(PROJECT_ROOT, "out", "compare")
FFMPEG = os.path.join(FLP_ROOT, "third_party", "ffmpeg-7.0.1-full_build", "bin", "ffmpeg.exe")
FPS = 25
LIP_IDX = [6, 12, 14, 17, 19, 20]

# ---------------------------------------------------------------- audio
with wave.open(WAV, "rb") as w:
    SR = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
DUR = len(pcm) / SR
N_FRAMES = int(DUR * FPS)


def rms_at(t, win=0.040):
    i0 = max(0, int((t - win / 2) * SR))
    i1 = min(len(pcm), int((t + win / 2) * SR))
    if i1 <= i0:
        return 0.0
    return float(np.sqrt(np.mean(pcm[i0:i1] ** 2)))


# ---------------------------------------------------------------- renderer (shared)
cfg = OmegaConf.load(os.path.join(FLP_ROOT, "configs", "trt_infer.yaml"))
pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
assert pipe.prepare_source(SOURCE, realtime=True)
(x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, _l0, _f0, mask_ori, M) = pipe.src_infos[0][0]
base_tensor = torch.from_numpy(pipe.src_imgs[0]).to(pipe.device).float()
c_s_eye = float(calc_eye_close_ratio(source_lmk[None])[0][:2].mean())

VIS = np.load(os.path.join(PROJECT_ROOT, "assets", "visemes.npy")).astype(x_s.dtype)  # ah, ee, oh

BLINKS = [1.3, 4.1]  # deterministic, same for all three
BLINK_DUR = 0.22


def blink_ratio(t):
    for b in BLINKS:
        if b <= t < b + BLINK_DUR:
            p = (t - b) / BLINK_DUR
            return c_s_eye - (c_s_eye - 0.03) * math.sin(math.pi * p)
    return None


def sway(t):
    dyaw = 2.0 * math.sin(2 * math.pi * t / 8.7) + 0.7 * math.sin(2 * math.pi * t / 3.1)
    dpitch = 1.1 * math.sin(2 * math.pi * t / 6.3)
    droll = 0.6 * math.sin(2 * math.pi * t / 11.2)
    return dyaw, dpitch, droll


def render_frame(mouth_delta, t):
    dyaw, dpitch, droll = sway(t)
    exp = x_s_info["exp"] + mouth_delta
    x_d = transform_keypoint(x_s_info["pitch"] + dpitch, x_s_info["yaw"] + dyaw,
                             x_s_info["roll"] + droll, x_s_info["t"], exp,
                             x_s_info["scale"], x_s_info["kp"])
    br = blink_ratio(t)
    if br is not None:
        comb_eye = pipe.calc_combined_eye_ratio([[br]], source_lmk)
        x_d = x_d + pipe.retarget_eye(x_s, comb_eye).reshape(-1, x_s.shape[1], 3)
    x_d = pipe.stitching(x_s, x_d)
    out = pipe.model_dict["warping_spade"].predict(f_s, x_s, x_d)
    out = paste_back_pytorch(out, M, base_tensor.clone(), mask_ori)
    return out.to(dtype=torch.uint8).cpu().numpy()


def write_video(name, mouth_track):
    tmp = os.path.join(OUT_DIR, name + "-noaudio.mp4")
    final = os.path.join(OUT_DIR, name + ".mp4")
    writer = None
    for i in range(N_FRAMES):
        frame = render_frame(mouth_track[i], i / FPS)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                     (bgr.shape[1], bgr.shape[0]))
        writer.write(bgr)
    writer.release()
    subprocess.run([FFMPEG, "-y", "-i", tmp, "-i", WAV, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final],
                   check=True, capture_output=True)
    os.remove(tmp)
    print("wrote", final)


# ---------------------------------------------------------------- A: envelope
def track_envelope(lip_gain=0.6):
    rng = np.random.RandomState(7)
    track, level, cur, prev = [], 0.0, 0, 0.0
    for i in range(N_FRAMES):
        t = i / FPS
        lvl = min(1.0, rms_at(t) / 0.12)
        level = max(lvl, level * (0.72 ** ((1 / FPS) / 0.05)))
        if prev < 0.30 <= level:
            cur = rng.choice(3, p=[0.5, 0.25, 0.25])
        prev = level
        track.append(VIS[cur] * ((level ** 1.2) * lip_gain))
    return track


# ---------------------------------------------------------------- B: fonemas
def g2p_es(text):
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn" or c == "̃")
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


PHONE_COST = {"a": 1.10, "e": 1.00, "i": 0.95, "o": 1.05, "u": 1.00,
              "m": 0.70, "b": 0.65, "p": 0.65, "f": 0.70,
              "s": 0.65, "x": 0.70, "tS": 0.75, "y": 0.65,
              "t": 0.55, "d": 0.55, "n": 0.55, "l": 0.55, "r": 0.50,
              "k": 0.55, "g": 0.55}
VISEME_OF = {"a": "A", "e": "EI", "i": "EI", "o": "OU", "u": "OU",
             "m": "MBP", "b": "MBP", "p": "MBP", "f": "FV"}
# template per viseme as a weighted mix of the mined shapes (ah, ee, oh)
TEMPLATES = {"A": VIS[0] * 0.85, "EI": VIS[1] * 0.75, "OU": VIS[2] * 0.95,
             "MBP": VIS[0] * -0.20, "FV": VIS[0] * 0.10, "C_NEUTRAL": VIS[0] * 0.25}


def speech_segments():
    hop = 0.01
    n = int(DUR / hop)
    e = np.array([rms_at(i * hop, win=0.03) for i in range(n)])
    thr = 0.10 * np.percentile(e, 98)
    on = e > thr
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i * hop
        elif not v and start is not None:
            if i * hop - start > 0.06:
                segs.append((start, i * hop))
            start = None
    if start is not None:
        segs.append((start, n * hop))
    return segs


def track_fonemas():
    phones = g2p_es(PHRASE)
    segs = speech_segments()
    speech_total = sum(b - a for a, b in segs)
    costs = [PHONE_COST.get(p, 0.60) for p in phones]
    unit = speech_total / sum(costs)

    # phone events in "speech time", then mapped to real time through the segments
    events, cursor = [], 0.0
    for p, c in zip(phones, costs):
        events.append((cursor, cursor + c * unit, VISEME_OF.get(p, "C_NEUTRAL")))
        cursor += c * unit

    def to_real(ts):
        acc = 0.0
        for a, b in segs:
            if ts <= acc + (b - a):
                return a + (ts - acc)
            acc += b - a
        return segs[-1][1]

    events = [(to_real(a), to_real(b), v) for a, b, v in events]

    track, prev = [], {k: 0.0 for k in TEMPLATES}
    for i in range(N_FRAMES):
        t = i / FPS
        raw = {k: 0.0 for k in TEMPLATES}
        for a, b, v in events:
            start = a - (0.040 if v in ("A", "EI", "OU") else 0.015)  # anticipation
            if t < start or t > b:
                continue
            center, half = 0.5 * (start + b), max(0.020, 0.5 * (b - start))
            w = math.exp(-0.5 * ((t - center) / half) ** 2)
            raw[v] = max(raw[v], w)
        s = sum(raw.values())
        if s > 1.0:
            raw = {k: v / s for k, v in raw.items()}
        out = {}
        for k in TEMPLATES:
            alpha = 0.45 if raw[k] > prev[k] else 0.20  # attack / release
            out[k] = prev[k] + alpha * (raw[k] - prev[k])
            out[k] = prev[k] + np.clip(out[k] - prev[k], -0.14, 0.14)
        prev = out
        delta = sum(TEMPLATES[k] * out[k] for k in TEMPLATES)
        track.append(delta.astype(x_s.dtype))
    return track


# ---------------------------------------------------------------- C: joyvasa lips only
def track_joyvasa():
    from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline
    jv = JoyVASAAudio2MotionPipeline(
        motion_model_path=cfg.joyvasa_models.motion_model_path,
        audio_model_path=cfg.joyvasa_models.audio_model_path,
        motion_template_path=cfg.joyvasa_models.motion_template_path,
        cfg_mode=cfg.infer_params.cfg_mode, cfg_scale=2.0)
    motion = jv.gen_motion_sequence(WAV)
    exps = [m["exp"] for m in motion["motion"]]
    ref = exps[0]
    track = []
    for i in range(N_FRAMES):
        j = min(i, len(exps) - 1)  # joyvasa fps == 25 == FPS
        delta = np.zeros_like(ref)
        delta[:, LIP_IDX, :] = exps[j][:, LIP_IDX, :] - ref[:, LIP_IDX, :]
        track.append(delta.astype(x_s.dtype))
    return track


print(f"audio {DUR:.1f}s -> {N_FRAMES} frames")
write_video("A_envelope", track_envelope())
write_video("B_fonemas", track_fonemas())
write_video("C_joyvasa_labios", track_joyvasa())
print("all done")
