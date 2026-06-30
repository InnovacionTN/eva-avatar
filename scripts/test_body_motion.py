"""Verify the body-motion warp: shoulders sway, feet stay planted, subtle.

Renders the body source through _body_warp at a neutral phase and a sway peak,
then measures the horizontal center-of-mass shift per row band. Also dumps two
full frames for visual inspection.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import eva_platica as ep  # noqa: E402

src = os.path.join(ep.PROJECT_ROOT, "assets", "eva_body.png")
r = ep.EvaRenderer(src, pasteback=True, body_motion=True, body_sway=1.0, body_breath=1.0)
assert r.body_motion, "body motion did not enable"

base = r.base_tensor.clone()  # HxWx3 float, the static body
H, W = r.bm_H, r.bm_W


def warp_np(tnow):
    out = r._body_warp(base.clone(), tnow, e=0.0, speaking=False, listening=False)
    return out.to(dtype=r.torch.uint8).cpu().numpy()


def band_com(img, y0f, y1f):
    """Horizontal center of mass of non-black luminance in a row band."""
    y0, y1 = int(y0f * H), int(y1f * H)
    band = img[y0:y1].astype(np.float32).mean(axis=2)  # rows x W
    mass = band.sum()
    if mass < 1e-3:
        return float("nan")
    xs = np.arange(W, dtype=np.float32)[None, :]
    return float((band * xs).sum() / mass)


a = warp_np(0.0)   # neutral phase
b = warp_np(2.0)   # near a sway peak

bands = {"head/shoulders 0.20-0.30": (0.20, 0.30),
         "chest 0.35-0.45": (0.35, 0.45),
         "hips 0.50-0.55": (0.50, 0.55),
         "knees 0.70-0.80": (0.70, 0.80),
         "feet 0.92-0.99": (0.92, 0.99)}
print(f"[test] image {W}x{H}  (1px = {2.0/W*100:.3f}% normalized)")
print("[test] horizontal shift between neutral and sway-peak, per band:")
for name, (y0, y1) in bands.items():
    shift = band_com(b, y0, y1) - band_com(a, y0, y1)
    print(f"  {name:28s}: {shift:+.2f} px")

out_dir = os.path.join(ep.PROJECT_ROOT, "out")
os.makedirs(out_dir, exist_ok=True)
for tag, im in (("neutral", a), ("swaypeak", b)):
    p = os.path.join(out_dir, f"body_motion_{tag}.png")
    cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    print(f"[test] saved {p}")
