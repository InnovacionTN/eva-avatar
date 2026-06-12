"""Real-time Eva: webcam -> LivePortrait -> virtual camera.

The operator's webcam drives the Eva portrait; the animated result is sent to
the OBS virtual camera so it can be selected in Meet/Teams/Zoom.

Usage:
    python src/run_realtime.py [--preview] [--no-mirror] [--fps-overlay]
                               [--camera-index 0] [--cam-width 1280] [--cam-height 720]
"""
import argparse
import os
import sys
import threading
import time

import config
from config import (LIVEPORTRAIT_DIR, SMOOTH_BETA, SMOOTH_MIN_CUTOFF,
                    SOURCE_IMAGE, TARGET_FPS)

sys.path.insert(0, LIVEPORTRAIT_DIR)

import cv2
import numpy as np
import torch

from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.camera import get_rotation_matrix
from src.utils.crop import (crop_image_by_bbox, parse_bbox_from_landmark,
                            paste_back, prepare_paste_back)
from src.utils.cropper import Cropper
from src.utils.helper import calc_motion_multiplier

from smoothing import OneEuroFilter


class WebcamReader(threading.Thread):
    """Reads frames on its own thread; main loop always gets the latest one."""

    def __init__(self, index, width, height, fps):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


def lmk_is_sane(lmk, w, h):
    if lmk is None or not np.isfinite(lmk).all():
        return False
    x0, y0 = lmk.min(0)
    x1, y1 = lmk.max(0)
    bw, bh = x1 - x0, y1 - y0
    return bw > 30 and bh > 30 and x1 > 0 and y1 > 0 and x0 < w and y0 < h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_IMAGE)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--out-width", type=int, default=1280)
    ap.add_argument("--out-height", type=int, default=720)
    ap.add_argument("--cam-fps", type=int, default=TARGET_FPS)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--preview", action="store_true", help="show a local preview window")
    ap.add_argument("--fps-overlay", action="store_true")
    ap.add_argument("--no-pasteback", action="store_true", help="output the raw 512 face crop")
    ap.add_argument("--driving-multiplier", type=float, default=1.0)
    ap.add_argument("--smooth-cutoff", type=float, default=SMOOTH_MIN_CUTOFF)
    ap.add_argument("--smooth-beta", type=float, default=SMOOTH_BETA)
    args = ap.parse_args()

    # ---- GPU pin guard ----
    name = torch.cuda.get_device_name(0)
    assert "RTX 2000 Ada" in name, f"inference landed on the wrong device: {name}"
    print(f"[init] device: {name}")

    # ---- models ----
    inf_cfg = InferenceConfig()  # half precision on by default
    crop_cfg = CropConfig()
    print("[init] loading LivePortrait...")
    wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
    cropper = Cropper(crop_cfg=crop_cfg)

    # ---- source: encode ONCE ----
    src_bgr = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if src_bgr is None:
        sys.exit(f"cannot read source image: {args.source}")
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    crop_info = cropper.crop_source_image(src_rgb, crop_cfg)
    if crop_info is None:
        sys.exit("no face detected in the source image")
    I_s = wrapper.prepare_source(crop_info["img_crop_256x256"])
    x_s_info = wrapper.get_kp_info(I_s)
    x_c_s = x_s_info["kp"]
    R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
    f_s = wrapper.extract_feature_3d(I_s)
    x_s = wrapper.transform_keypoint(x_s_info)
    mask_ori_float = prepare_paste_back(
        inf_cfg.mask_crop, crop_info["M_c2o"],
        dsize=(src_rgb.shape[1], src_rgb.shape[0]))
    print("[init] source encoded")

    # ---- warmup + VRAM log ----
    for _ in range(2):
        out = wrapper.warp_decode(f_s, x_s, x_s)
    wrapper.parse_output(out["out"])
    torch.cuda.synchronize()
    print(f"[init] warmup done, VRAM allocated: "
          f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB "
          f"(reserved {torch.cuda.memory_reserved(0) / 1e9:.2f} GB)")

    # ---- output canvas (static background from the source portrait) ----
    OW, OH = args.out_width, args.out_height
    side = min(OW, OH)
    x_off, y_off = (OW - side) // 2, (OH - side) // 2
    bg = cv2.resize(src_rgb, (OW, OH))
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    canvas_base = bg.copy()

    # ---- smoothing filters on driving params ----
    fk = dict(min_cutoff=args.smooth_cutoff, beta=args.smooth_beta)
    filt = {k: OneEuroFilter(**fk) for k in ("pitch", "yaw", "roll", "t", "scale", "exp")}

    # ---- webcam + virtual camera ----
    import pyvirtualcam
    reader = WebcamReader(args.camera_index, args.cam_width, args.cam_height, args.cam_fps)
    reader.start()
    t0 = time.time()
    while reader.latest() is None:
        if time.time() - t0 > 10:
            sys.exit("webcam produced no frames in 10s")
        time.sleep(0.05)
    print("[init] webcam streaming")

    cam = pyvirtualcam.Camera(width=OW, height=OH, fps=args.cam_fps,
                              fmt=pyvirtualcam.PixelFormat.RGB)
    print(f"[init] virtual camera: {cam.device} {OW}x{OH}@{args.cam_fps}")

    # ---- tracking / driving state ----
    lmk_prev = None
    bbox_ema = None
    R_d_0 = x_d_0_exp = x_d_0_scale = x_d_0_t = None
    x_d_0_new = motion_multiplier = None
    calib_hist = []  # (t, lmk_center, yaw_deg, pitch_deg) while waiting to anchor
    last_frame_rgb = None  # hold-last-pose output
    n_done, n_lost, fps_now = 0, 0, 0.0
    fps_t0, fps_n0 = time.time(), 0

    # neutral output shown until the operator pose is calibrated
    neutral_canvas = canvas_base.copy()
    neutral_canvas[y_off:y_off + side, x_off:x_off + side] = cv2.resize(src_rgb, (side, side))

    print("[run] live. Ctrl+C to stop." + (" Press q in preview to quit." if args.preview else ""))
    try:
        while True:
            frame = reader.latest()
            if frame is None:
                time.sleep(0.005)
                continue
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # -- track or (re)detect the operator's face --
            lmk = None
            if lmk_prev is not None:
                lmk = cropper.human_landmark_runner.run(rgb, lmk_prev)
                if not lmk_is_sane(lmk, w, h):
                    lmk = None
            if lmk is None:
                n_lost += 1
                if n_lost % 10 == 1:  # retry detection ~every 10 lost frames
                    faces = cropper.face_analysis_wrapper.get(
                        np.ascontiguousarray(rgb[..., ::-1]),
                        flag_do_landmark_2d_106=True, direction="large-small")
                    if faces:
                        lmk = cropper.human_landmark_runner.run(rgb, faces[0].landmark_2d_106)
                        if not lmk_is_sane(lmk, w, h):
                            lmk = None
                if lmk is None:
                    # hold the last good pose; never snap to neutral
                    if last_frame_rgb is not None:
                        cam.send(last_frame_rgb)
                    if args.preview and last_frame_rgb is not None:
                        cv2.imshow("Eva", cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue
                # recovered: drop the anchor and recalibrate from a fresh frontal pose
                for f in filt.values():
                    f.reset()
                bbox_ema = None
                R_d_0 = motion_multiplier = None
                calib_hist = []
                print("[run] face reacquired - recalibrating...")
            n_lost = 0
            lmk_prev = lmk

            # -- crop the driving face --
            ret_bbox = parse_bbox_from_landmark(
                lmk, scale=crop_cfg.scale_crop_driving_video,
                vx_ratio_crop_driving_video=crop_cfg.vx_ratio_crop_driving_video,
                vy_ratio=crop_cfg.vy_ratio_crop_driving_video)["bbox"]
            bbox = np.array([ret_bbox[0, 0], ret_bbox[0, 1], ret_bbox[2, 0], ret_bbox[2, 1]])
            bbox_ema = bbox if bbox_ema is None else 0.9 * bbox_ema + 0.1 * bbox
            crop512 = crop_image_by_bbox(rgb, list(bbox_ema), lmk=lmk,
                                         dsize=512, flag_rot=False, borderValue=(0, 0, 0))
            crop256 = cv2.resize(crop512["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)

            # -- driving motion --
            I_d = wrapper.prepare_source(crop256)
            x_d_info = wrapper.get_kp_info(I_d)
            tnow = time.time()

            if R_d_0 is None:
                # calibration gate: only anchor on a frontal, still pose, so a
                # bad startup posture can't skew all subsequent motion
                yaw_deg = float(x_d_info["yaw"].mean())
                pitch_deg = float(x_d_info["pitch"].mean())
                center = lmk.mean(0)
                calib_hist.append((tnow, center, yaw_deg, pitch_deg))
                calib_hist = [c for c in calib_hist if tnow - c[0] <= 0.8]
                centers = np.stack([c[1] for c in calib_hist])
                stable = (len(calib_hist) >= 3
                          and all(abs(c[2]) < 15 and abs(c[3]) < 12 for c in calib_hist)
                          and np.ptp(centers, axis=0).max() < 8)
                if not stable:
                    cam.send(neutral_canvas)
                    last_frame_rgb = neutral_canvas
                    if args.preview:
                        cv2.imshow("Eva", cv2.cvtColor(neutral_canvas, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue
                print("[run] pose calibrated - driving live")

            pitch = filt["pitch"](x_d_info["pitch"], tnow)
            yaw = filt["yaw"](x_d_info["yaw"], tnow)
            roll = filt["roll"](x_d_info["roll"], tnow)
            t_d = filt["t"](x_d_info["t"], tnow)
            scale_d = filt["scale"](x_d_info["scale"], tnow)
            exp_d = filt["exp"](x_d_info["exp"], tnow)
            R_d = get_rotation_matrix(pitch, yaw, roll)

            if R_d_0 is None:  # anchor frame for relative motion
                R_d_0 = R_d.clone()
                x_d_0_exp, x_d_0_scale, x_d_0_t = exp_d.clone(), scale_d.clone(), t_d.clone()

            delta_new = x_s_info["exp"] + (exp_d - x_d_0_exp)
            R_new = (R_d @ R_d_0.permute(0, 2, 1)) @ R_s
            scale_new = x_s_info["scale"] * (scale_d / x_d_0_scale)
            t_new = x_s_info["t"] + (t_d - x_d_0_t)
            t_new[..., 2].fill_(0)
            x_d_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if motion_multiplier is None:  # expression-friendly driving
                x_d_0_new = x_d_new.clone()
                motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
            x_d_new = (x_d_new - x_d_0_new) * motion_multiplier + x_s

            x_d_new = wrapper.stitching(x_s, x_d_new)
            x_d_new = x_s + (x_d_new - x_s) * args.driving_multiplier

            out = wrapper.warp_decode(f_s, x_s, x_d_new)
            I_p = wrapper.parse_output(out["out"])[0]  # 512x512 RGB

            # -- compose output frame --
            if args.no_pasteback:
                face = I_p
            else:
                face = paste_back(I_p, crop_info["M_c2o"], src_rgb, mask_ori_float)
            canvas = canvas_base.copy()
            canvas[y_off:y_off + side, x_off:x_off + side] = cv2.resize(face, (side, side))

            if args.fps_overlay:
                cv2.putText(canvas, f"{fps_now:.1f} fps", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cam.send(canvas)
            last_frame_rgb = canvas

            if args.preview:
                cv2.imshow("Eva", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # -- fps bookkeeping --
            n_done += 1
            if n_done - fps_n0 >= 30:
                tnow2 = time.time()
                fps_now = (n_done - fps_n0) / (tnow2 - fps_t0)
                fps_t0, fps_n0 = tnow2, n_done
                vram = torch.cuda.memory_allocated(0) / 1e9
                print(f"[run] {fps_now:.1f} fps | VRAM {vram:.2f} GB | frames {n_done}")
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        cam.close()
        if args.preview:
            cv2.destroyAllWindows()
        print(f"[done] {n_done} frames")


if __name__ == "__main__":
    main()
