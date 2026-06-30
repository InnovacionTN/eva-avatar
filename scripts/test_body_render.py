"""Smoke test: render the full-body Eva source once and save it.

Confirms FasterLivePortrait detects the (small) face on the standing-body
image, crops/animates it, and pastes it back onto the full body.
Run with the FLP package python.
"""
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import eva_platica as ep  # noqa: E402

src = os.path.join(ep.PROJECT_ROOT, "assets", "eva_body.png")
print(f"[test] source: {src}")
r = ep.EvaRenderer(src, pasteback=True, lip_gain=1.25, sway=1.0)
print(f"[test] src_rgb shape: {r.src_rgb.shape}")

frame = r.frame(0.6, speaking=True)  # mid-open mouth
print(f"[test] rendered frame: {frame.shape}")
out_dir = os.path.join(ep.PROJECT_ROOT, "out")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "body_render_test.png")
cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
print(f"[test] saved: {out_path}")
