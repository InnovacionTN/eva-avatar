"""Offline reenactment: eva_source.png + driving video -> output video.

Wraps third_party/LivePortrait/inference.py. Reports achieved processing fps.

Usage:
    python src/run_offline.py [--driving PATH] [--output-dir PATH]
                              [--no-stitching] [--eye-retargeting] [--lip-retargeting]
"""
import argparse
import os
import subprocess
import sys
import time

import config
from config import LIVEPORTRAIT_DIR, PROJECT_ROOT, SOURCE_IMAGE

DEFAULT_DRIVING = os.path.join(LIVEPORTRAIT_DIR, "assets", "examples", "driving", "d0.mp4")


def count_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_IMAGE)
    ap.add_argument("--driving", default=DEFAULT_DRIVING)
    ap.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "out"))
    ap.add_argument("--no-stitching", action="store_true")
    ap.add_argument("--eye-retargeting", action="store_true")
    ap.add_argument("--lip-retargeting", action="store_true")
    ap.add_argument("--driving-multiplier", type=float, default=1.0)
    args = ap.parse_args()

    for p in (args.source, args.driving):
        if not os.path.isfile(p):
            sys.exit(f"missing input: {p}")
    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(LIVEPORTRAIT_DIR, "inference.py"),
        "-s", args.source,
        "-d", args.driving,
        "-o", args.output_dir,
        "--device-id", "0",
        "--driving-multiplier", str(args.driving_multiplier),
    ]
    if args.no_stitching:
        cmd.append("--no-flag-stitching")
    if args.eye_retargeting:
        cmd.append("--flag-eye-retargeting")
    if args.lip_retargeting:
        cmd.append("--flag-lip-retargeting")

    # inference.py builds its own onnxruntime sessions; give it torch's CUDA DLLs too.
    # PYTHONUTF8: rich's progress bar prints emoji that cp1252 consoles can't encode.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUTF8="1",
               PATH=config.TORCH_LIB + os.pathsep + os.environ.get("PATH", ""))
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=LIVEPORTRAIT_DIR, env=env).returncode
    elapsed = time.time() - t0
    if ret != 0:
        sys.exit(f"inference failed (exit {ret})")

    frames = count_frames(args.driving)
    print(f"\nprocessed {frames} driving frames in {elapsed:.1f}s "
          f"(~{frames / elapsed:.1f} fps end-to-end, incl. model load)")
    print(f"output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
