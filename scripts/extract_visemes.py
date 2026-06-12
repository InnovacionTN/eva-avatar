"""Mine real speech mouth shapes (visemes) from the FLP example videos.

retarget_lip only spreads the lips, which reads as open-lips/clenched-teeth.
Real talking needs the jaw + mouth interior, which live in the expression
deltas of actual speech frames. This scans the driving videos, picks the
single most expressive talker, and saves three mouth shapes relative to that
person's closed mouth: 'ah' (max open), 'ee' (wide), 'oh' (narrow).
Output: assets/visemes.npy with shape (3, 1, 21, 3), lip keypoints only.
"""
import glob
import os
import sys

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
from src.utils.crop import parse_bbox_from_landmark, crop_image_by_bbox
from src.utils.utils import calc_lip_close_ratio

VIDEOS = sorted(glob.glob(os.path.join(FLP_ROOT, "assets", "examples", "driving", "d*.mp4")))
LIP_IDX = [6, 12, 14, 17, 19, 20]

cfg = OmegaConf.load(os.path.join(FLP_ROOT, "configs", "trt_infer.yaml"))
pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)

by_video = {}
for vid in VIDEOS:
    cap = cv2.VideoCapture(vid)
    lmk_pre = None
    idx = 0
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 2 == 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if lmk_pre is None:
            faces = pipe.model_dict["face_analysis"].predict(frame)
            if not faces:
                continue
            lmk_pre = pipe.model_dict["landmark"].predict(rgb, faces[0])
        else:
            lmk_pre = pipe.model_dict["landmark"].predict(rgb, lmk_pre)
        bbox = parse_bbox_from_landmark(lmk_pre, scale=cfg.crop_params.dri_scale,
                                        vx_ratio_crop_video=cfg.crop_params.dri_vx_ratio,
                                        vy_ratio=cfg.crop_params.dri_vy_ratio)["bbox"]
        ret = crop_image_by_bbox(rgb, [bbox[0, 0], bbox[0, 1], bbox[2, 0], bbox[2, 1]],
                                 lmk=lmk_pre, dsize=512, flag_rot=False, borderValue=(0, 0, 0))
        lmk = ret["lmk_crop"]
        crop256 = cv2.resize(ret["img_crop"], (256, 256))

        eye_l = lmk[0:24].mean(0)
        eye_r = lmk[24:48].mean(0)
        face_scale = float(np.linalg.norm(eye_l - eye_r))
        if face_scale < 1:
            continue
        width = float(np.linalg.norm(lmk[48] - lmk[66])) / face_scale
        openness = float(calc_lip_close_ratio(lmk[None])[0][0])
        _, _, _, _, exp, _, _ = pipe.model_dict["motion_extractor"].predict(crop256)
        rows.append((openness, width, exp.copy()))
    cap.release()
    if len(rows) >= 40:
        by_video[os.path.basename(vid)] = rows
    rng = max((r[0] for r in rows), default=0) - min((r[0] for r in rows), default=0)
    print(f"{os.path.basename(vid)}: {len(rows)} frames, openness range {rng:.3f}")

# the best source: the video with the largest openness range (expressive talker)
best = max(by_video, key=lambda v: np.ptp([r[0] for r in by_video[v]]))
rows = by_video[best]
opens = np.array([r[0] for r in rows])
widths = np.array([r[1] for r in rows])
print(f"using {best} ({len(rows)} frames)")

closed_i = int(np.argmin(np.abs(opens - np.percentile(opens, 10))))
neutral_exp = rows[closed_i][2]

ah_i = int(np.argmax(opens))  # widest open jaw
mid = (opens > np.percentile(opens, 55)) & (opens < np.percentile(opens, 85))
mid_rows = [i for i in range(len(rows)) if mid[i]]
ee_i = max(mid_rows, key=lambda i: widths[i])   # mid-open, widest (ee)
oh_i = min(mid_rows, key=lambda i: widths[i])   # mid-open, narrowest (oh)

names = ["ah", "ee", "oh"]
visemes = []
for n, i in zip(names, (ah_i, ee_i, oh_i)):
    delta = np.zeros_like(neutral_exp)
    delta[:, LIP_IDX, :] = rows[i][2][:, LIP_IDX, :] - neutral_exp[:, LIP_IDX, :]
    visemes.append(delta)
    print(f"{n}: open {opens[i]:.3f} width {widths[i]:.3f} |delta| {np.abs(delta).sum():.4f}")

out_path = os.path.join(PROJECT_ROOT, "assets", "visemes.npy")
np.save(out_path, np.stack(visemes))
print(f"saved {out_path}")
