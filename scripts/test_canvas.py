"""Render the final 1280x720 letterboxed canvas (full body on black)."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import eva_platica as ep  # noqa: E402

src = os.path.join(ep.PROJECT_ROOT, "assets", "eva_body.png")
r = ep.EvaRenderer(src, pasteback=True, lip_gain=1.25, body_motion=True)

OW, OH = 1280, 720
fh, fw = r.src_rgb.shape[:2]
scale = min(OW / fw, OH / fh)
dst_w, dst_h = int(round(fw * scale)), int(round(fh * scale))
x_off, y_off = (OW - dst_w) // 2, (OH - dst_h) // 2
print(f"[test] src {fw}x{fh} -> dst {dst_w}x{dst_h} at ({x_off},{y_off}) in {OW}x{OH}")

face = r.frame(0.6, speaking=True)
canvas = np.zeros((OH, OW, 3), dtype=np.uint8)
canvas[y_off:y_off + dst_h, x_off:x_off + dst_w] = cv2.resize(face, (dst_w, dst_h))

out_path = os.path.join(ep.PROJECT_ROOT, "out", "canvas_16x9.png")
cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print(f"[test] saved {out_path}")
