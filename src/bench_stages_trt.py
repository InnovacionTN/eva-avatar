"""Per-stage latency benchmark of the FasterLivePortrait TRT pipeline."""
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CAND = glob.glob(os.path.join(PROJECT_ROOT, "third_party", "FLP-win", "**", "configs", "trt_infer.yaml"),
                 recursive=True)
FLP_ROOT = os.path.dirname(os.path.dirname(CAND[0]))
os.chdir(FLP_ROOT)
sys.path.insert(0, FLP_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

SOURCE = os.path.join(PROJECT_ROOT, "assets", "eva_source.png")
N = 50

cfg = OmegaConf.load(os.path.join(FLP_ROOT, "configs", "trt_infer.yaml"))
cfg.infer_params.flag_crop_driving_video = True
pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
assert pipe.prepare_source(SOURCE, realtime=True)

src_rgb = pipe.src_imgs[0]
frame = cv2.cvtColor(cv2.resize(src_rgb, (1280, 720)), cv2.COLOR_RGB2BGR)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

faces = pipe.model_dict["face_analysis"].predict(frame)
lmk0 = pipe.model_dict["landmark"].predict(rgb, faces[0])
crop256 = cv2.resize(rgb, (256, 256))


def bench(label, fn, n=N):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{label:24s} {ms:7.1f} ms")
    return ms


bench("landmark (trt)", lambda: pipe.model_dict["landmark"].predict(rgb, lmk0))
bench("motion_extractor (trt)", lambda: pipe.model_dict["motion_extractor"].predict(crop256))
x_s_info = pipe.src_infos[0][0][0]
f_s = pipe.src_infos[0][0][3]
x_s = pipe.src_infos[0][0][4]
bench("stitching (trt)", lambda: pipe.stitching(x_s, x_s))
bench("warping_spade (trt)", lambda: pipe.model_dict["warping_spade"].predict(f_s, x_s, x_s))
bench("face_analysis (trt)", lambda: pipe.model_dict["face_analysis"].predict(frame))

# full per-frame call as used in run_realtime_trt
pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0], first_frame=True)
bench("pipe.run() full", lambda: pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0], first_frame=False))
