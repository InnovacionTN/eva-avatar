"""Offline JoyVASA lip-sync test: saved Gemini reply -> Eva video with audio.

Generates LivePortrait motion from out/gemini_check.wav with JoyVASA, renders
it through the TRT pipeline onto the Eva source, muxes the audio, and reports
generation speed (the latency cost of using this live).
Output: out/joyvasa_test.mp4
"""
import glob
import os
import subprocess
import sys
import time

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

from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.pipelines.joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline

WAV = os.path.join(PROJECT_ROOT, "out", "gemini_check.wav")
SOURCE = os.path.join(PROJECT_ROOT, "assets", "eva_source.png")
OUT_MP4 = os.path.join(PROJECT_ROOT, "out", "joyvasa_test.mp4")
FFMPEG = os.path.join(FLP_ROOT, "third_party", "ffmpeg-7.0.1-full_build", "bin", "ffmpeg.exe")

cfg = OmegaConf.load(os.path.join(FLP_ROOT, "configs", "trt_infer.yaml"))

print("[1/3] generating motion from audio (JoyVASA)...")
jv = JoyVASAAudio2MotionPipeline(
    motion_model_path=cfg.joyvasa_models.motion_model_path,
    audio_model_path=cfg.joyvasa_models.audio_model_path,
    motion_template_path=cfg.joyvasa_models.motion_template_path,
    cfg_mode=cfg.infer_params.cfg_mode, cfg_scale=cfg.infer_params.cfg_scale)
t0 = time.time()
motion = jv.gen_motion_sequence(WAV)
gen_s = time.time() - t0
import wave
with wave.open(WAV, "rb") as w:
    audio_s = w.getnframes() / w.getframerate()
n = motion["n_frames"]
fps = motion["output_fps"]
print(f"audio {audio_s:.1f}s -> {n} frames @ {fps} fps | generation took {gen_s:.1f}s "
      f"({gen_s / audio_s:.2f}x realtime)")

print("[2/3] rendering through TRT pipeline...")
pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
assert pipe.prepare_source(SOURCE, realtime=True)
tmp_mp4 = OUT_MP4.replace(".mp4", "-noaudio.mp4")
writer = None
t0 = time.time()
for i in range(n):
    out_crop, out_org = pipe.run_with_pkl([motion["motion"][i], None, None],
                                          pipe.src_imgs[0], pipe.src_infos[0],
                                          first_frame=(i == 0))
    frame = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
    if writer is None:
        writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (frame.shape[1], frame.shape[0]))
    writer.write(frame)
writer.release()
render_s = time.time() - t0
print(f"rendered {n} frames in {render_s:.1f}s ({n / render_s:.1f} fps)")

print("[3/3] muxing audio...")
subprocess.run([FFMPEG, "-y", "-i", tmp_mp4, "-i", WAV, "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", OUT_MP4],
               check=True, capture_output=True)
os.remove(tmp_mp4)
print(f"done: {OUT_MP4}")
