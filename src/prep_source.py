"""Prep and validate the Eva source portrait.

Loads assets/eva_source_raw.png, validates it (exactly one face, roughly
frontal, face reasonably centered), center-crops to a square, resizes to
512x512 and saves assets/eva_source.png.
"""
import os
import sys

import config  # noqa: F401  (pins CUDA_VISIBLE_DEVICES before onnxruntime/torch load)
from config import LIVEPORTRAIT_DIR, SOURCE_IMAGE, SOURCE_IMAGE_RAW

sys.path.insert(0, LIVEPORTRAIT_DIR)

import cv2
import numpy as np
from src.utils.dependencies.insightface.app import FaceAnalysis


def fail(msg):
    print(f"VALIDATION FAILED: {msg}")
    sys.exit(1)


def main():
    if not os.path.isfile(SOURCE_IMAGE_RAW):
        fail(f"missing source image: {SOURCE_IMAGE_RAW}")

    img = cv2.imread(SOURCE_IMAGE_RAW, cv2.IMREAD_COLOR)
    if img is None:
        fail("could not decode the source image")
    h, w = img.shape[:2]

    # center-crop to square, then resize to 512x512
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    img = img[y0:y0 + side, x0:x0 + side]
    if side != 512:
        interp = cv2.INTER_AREA if side > 512 else cv2.INTER_CUBIC
        img = cv2.resize(img, (512, 512), interpolation=interp)

    insightface_root = os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights", "insightface")
    app = FaceAnalysis(name="buffalo_l", root=insightface_root,
                       allowed_modules=["detection"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))
    faces = app.get(img)

    if len(faces) == 0:
        fail("no face detected")
    if len(faces) > 1:
        fail(f"{len(faces)} faces detected, expected exactly 1")

    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    fw, fh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if fw < 80 or fh < 80:
        fail(f"face too small ({fw:.0f}x{fh:.0f}px) — use a tighter portrait")
    if abs(cx - 256) > 100:
        fail(f"face is off-center horizontally (center x={cx:.0f})")

    # frontal check: nose x should sit between the eyes, roughly midway
    le, re, nose = face.kps[0], face.kps[1], face.kps[2]
    eye_span = re[0] - le[0]
    if eye_span <= 0 or not (0.25 < (nose[0] - le[0]) / eye_span < 0.75):
        fail("face does not look frontal (nose offset vs eyes)")

    cv2.imwrite(SOURCE_IMAGE, img)
    print(f"OK: 1 frontal face (det score {face.det_score:.2f}, "
          f"bbox {fw:.0f}x{fh:.0f}, center {cx:.0f},{cy:.0f})")
    print(f"saved: {SOURCE_IMAGE}")


if __name__ == "__main__":
    main()
