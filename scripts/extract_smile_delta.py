"""Mine a natural closed-lip smile from the FLP example driving videos.

Scans driving videos with the TRT landmark + motion extractor, scores each
frame by mouth-corner spread (normalized by inter-eye distance) while
penalizing open lips, then saves the expression delta (smile - neutral,
lip keypoints only) to assets/smile_delta.npy for eva_platica.py.
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
# pipeline's "lip" animation region: the exp keypoints that shape the mouth
LIP_IDX = [6, 12, 14, 17, 19, 20]

cfg = OmegaConf.load(os.path.join(FLP_ROOT, "configs", "trt_infer.yaml"))
pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)

records = []  # (smile_score, openness, exp, video, frame_idx)
for vid in VIDEOS:
    cap = cv2.VideoCapture(vid)
    lmk_pre = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 2 == 0:  # every other frame is plenty
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

        mouth_w = float(np.linalg.norm(lmk[48] - lmk[66]))
        eye_l = lmk[0:24].mean(0)
        eye_r = lmk[24:48].mean(0)
        face_scale = float(np.linalg.norm(eye_l - eye_r))
        if face_scale < 1:
            continue
        width_ratio = mouth_w / face_scale
        openness = float(calc_lip_close_ratio(lmk[None])[0][0])

        _, _, _, _, exp, _, _ = pipe.model_dict["motion_extractor"].predict(crop256)
        records.append((width_ratio, openness, exp.copy(), os.path.basename(vid), idx))
    cap.release()
    print(f"{os.path.basename(vid)}: scanned, total records {len(records)}")

if len(records) < 50:
    sys.exit("not enough faces found")

widths = np.array([r[0] for r in records])
opens = np.array([r[1] for r in records])

# closed-lip smile: widest mouth among frames with below-median openness
closed = [r for r in records if r[1] <= np.percentile(opens, 50)]
smile = max(closed, key=lambda r: r[0])
# neutral: most median width among closed-lip frames
med_w = float(np.median([r[0] for r in closed]))
neutral = min(closed, key=lambda r: abs(r[0] - med_w))

print(f"smile:   {smile[3]} frame {smile[4]} width {smile[0]:.3f} open {smile[1]:.3f}")
print(f"neutral: {neutral[3]} frame {neutral[4]} width {neutral[0]:.3f} open {neutral[1]:.3f}")

delta = np.zeros_like(smile[2])  # (1, 21, 3)
delta[:, LIP_IDX, :] = smile[2][:, LIP_IDX, :] - neutral[2][:, LIP_IDX, :]
out_path = os.path.join(PROJECT_ROOT, "assets", "smile_delta.npy")
np.save(out_path, delta)
print(f"saved {out_path} | delta magnitude {np.abs(delta).sum():.4f}")
