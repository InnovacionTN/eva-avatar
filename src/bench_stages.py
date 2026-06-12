"""Per-stage latency benchmark of the realtime pipeline (no webcam needed)."""
import os
import sys
import time

import config
from config import LIVEPORTRAIT_DIR, SOURCE_IMAGE

sys.path.insert(0, LIVEPORTRAIT_DIR)

import cv2
import numpy as np
import torch

cudnn_bench = "--cudnn-benchmark" in sys.argv
if cudnn_bench:
    torch.backends.cudnn.benchmark = True

from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.camera import get_rotation_matrix
from src.utils.cropper import Cropper

N = 50

inf_cfg = InferenceConfig()
crop_cfg = CropConfig()
wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
cropper = Cropper(crop_cfg=crop_cfg)

src_rgb = cv2.cvtColor(cv2.imread(SOURCE_IMAGE), cv2.COLOR_BGR2RGB)
crop_info = cropper.crop_source_image(src_rgb, crop_cfg)
I_s = wrapper.prepare_source(crop_info["img_crop_256x256"])
x_s_info = wrapper.get_kp_info(I_s)
R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
f_s = wrapper.extract_feature_3d(I_s)
x_s = wrapper.transform_keypoint(x_s_info)

frame = cv2.resize(src_rgb, (1280, 720))
crop256 = crop_info["img_crop_256x256"]
lmk = crop_info["lmk_crop"]

def bench(label, fn, n=N):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{label:24s} {ms:7.1f} ms")
    return ms

print(f"cudnn.benchmark = {torch.backends.cudnn.benchmark}")
total = 0.0
total += bench("landmark_runner", lambda: cropper.human_landmark_runner.run(frame, lmk))
total += bench("prepare_source", lambda: wrapper.prepare_source(crop256))
I_d = wrapper.prepare_source(crop256)
total += bench("get_kp_info", lambda: wrapper.get_kp_info(I_d))
total += bench("stitching", lambda: wrapper.stitching(x_s, x_s))
total += bench("warp_decode", lambda: wrapper.warp_decode(f_s, x_s, x_s))
out = wrapper.warp_decode(f_s, x_s, x_s)
total += bench("parse_output", lambda: wrapper.parse_output(out["out"]))

from src.utils.crop import paste_back, prepare_paste_back
mask = prepare_paste_back(inf_cfg.mask_crop, crop_info["M_c2o"],
                          dsize=(src_rgb.shape[1], src_rgb.shape[0]))
I_p = wrapper.parse_output(out["out"])[0]
total += bench("paste_back", lambda: paste_back(I_p, crop_info["M_c2o"], src_rgb, mask))
total += bench("canvas+resize", lambda: cv2.resize(I_p, (720, 720)))

print(f"{'TOTAL':24s} {total:7.1f} ms  -> {1000/total:.1f} fps theoretical")
