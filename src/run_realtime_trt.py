"""Real-time Eva on TensorRT: webcam -> FasterLivePortrait -> virtual camera.

Runs inside the FasterLivePortrait integrated package environment (its python
has tensorrt + the grid_sample plugin preinstalled). Keeps the same operator
experience as run_realtime.py: pose-calibration gate, face-loss hold,
driving-param smoothing, OBS virtual camera output.

Usage (from the package python):
    python run_realtime_trt.py [--preview] [--no-mirror] [--fps-overlay]
    python run_realtime_trt.py --driving-video path.mp4 --max-frames 200  # headless test
"""
import argparse
import copy
import glob
import os
import sys
import threading
import time

import cv2
import numpy as np

# ---- locate the FasterLivePortrait package root ----
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CANDIDATES = glob.glob(os.path.join(PROJECT_ROOT, "third_party", "FLP-win", "**", "configs", "trt_infer.yaml"),
                       recursive=True)
if not CANDIDATES:
    sys.exit("FasterLivePortrait package not found under third_party/FLP-win")
FLP_ROOT = os.path.dirname(os.path.dirname(CANDIDATES[0]))
os.chdir(FLP_ROOT)          # model paths in the yaml are relative to the package root
sys.path.insert(0, FLP_ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils import utils as flp_utils

SOURCE_IMAGE = os.path.join(PROJECT_ROOT, "assets", "eva_source.png")


class SmoothedPipeline(FasterLivePortraitPipeline):
    """The stock pipeline only smooths video sources; webcam driving (image
    source) gets raw per-frame motion. Smooth R/exp/t/scale before _run so the
    relative-motion math sees filtered params. Filters reset with the anchor."""

    def __init__(self, cfg, smooth_cutoff=4.0, smooth_beta=0.3, **kwargs):
        self._fk = dict(mincutoff=smooth_cutoff, beta=smooth_beta)
        super().__init__(cfg, **kwargs)

    def init_vars(self, **kwargs):
        super().init_vars(**kwargs)
        self._reset_filters()

    def _reset_filters(self):
        self.R_d_smooth = flp_utils.OneEuroFilter(**self._fk)
        self.exp_smooth = flp_utils.OneEuroFilter(**self._fk)
        self._t_smooth = flp_utils.OneEuroFilter(**self._fk)
        self._scale_smooth = flp_utils.OneEuroFilter(**self._fk)

    def run(self, image, img_src, src_info, **kwargs):
        if kwargs.get("first_frame", False) or self.R_d_0 is None:
            self._reset_filters()
        out = super().run(image, img_src, src_info, **kwargs)
        # super().run resets R_d_smooth/exp_smooth with its own constants on
        # the anchor frame; ours must win
        if kwargs.get("first_frame", False):
            self._reset_filters()
        return out

    def _run(self, src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, *args, **kwargs):
        if not self.is_source_video:  # image source: stock path is unsmoothed
            R_d_i = self.R_d_smooth.process(R_d_i)
            x_d_i_info = copy.copy(x_d_i_info)
            x_d_i_info["exp"] = self.exp_smooth.process(x_d_i_info["exp"])
            x_d_i_info["t"] = self._t_smooth.process(x_d_i_info["t"])
            x_d_i_info["scale"] = self._scale_smooth.process(x_d_i_info["scale"])
        return super()._run(src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, *args, **kwargs)


class WebcamReader(threading.Thread):
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


class VideoFileReader:
    """Drop-in frame source for headless testing: loops a driving video."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit(f"cannot open driving video: {path}")

    def start(self):
        pass

    def latest(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self):
        self.cap.release()


def lmk_is_sane(lmk, w, h):
    if lmk is None or not np.isfinite(lmk).all():
        return False
    x0, y0 = lmk.min(0)
    x1, y1 = lmk.max(0)
    return (x1 - x0) > 30 and (y1 - y0) > 30 and x1 > 0 and y1 > 0 and x0 < w and y0 < h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE_IMAGE)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--out-width", type=int, default=1280)
    ap.add_argument("--out-height", type=int, default=720)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--fps-overlay", action="store_true")
    ap.add_argument("--no-pasteback", action="store_true",
                    help="output the raw 512 face crop (saves ~10 ms/frame)")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--smooth-cutoff", type=float, default=4.0)
    ap.add_argument("--smooth-beta", type=float, default=0.3)
    ap.add_argument("--driving-multiplier", type=float, default=1.0)
    ap.add_argument("--driving-video", default=None,
                    help="drive from a video file instead of the webcam (testing)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="exit after N output frames (testing)")
    ap.add_argument("--no-virtual-cam", action="store_true",
                    help="skip pyvirtualcam output (testing)")
    args = ap.parse_args()

    print(f"[init] package root: {FLP_ROOT}")
    cfg = OmegaConf.load(os.path.join(FLP_ROOT, "configs", "trt_infer.yaml"))
    cfg.infer_params.flag_crop_driving_video = True  # webcam frames need face cropping
    cfg.infer_params.driving_multiplier = args.driving_multiplier
    # pasteback is gated by this flag (cannot pass realtime= to run(): the
    # package forwards it twice into _run and TypeErrors)
    use_pasteback = not args.no_pasteback
    cfg.infer_params.flag_pasteback = use_pasteback

    print("[init] loading TensorRT engines...")
    if args.no_smooth:
        pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
    else:
        pipe = SmoothedPipeline(cfg=cfg, smooth_cutoff=args.smooth_cutoff,
                                smooth_beta=args.smooth_beta, is_animal=False)
    if not pipe.prepare_source(args.source, realtime=True):
        sys.exit(f"no face detected in source: {args.source}")
    print("[init] source encoded")

    # ---- output canvas ----
    OW, OH = args.out_width, args.out_height
    side = min(OW, OH)
    x_off, y_off = (OW - side) // 2, (OH - side) // 2
    src_rgb = pipe.src_imgs[0]
    bg = cv2.GaussianBlur(cv2.resize(src_rgb, (OW, OH)), (0, 0), 25)
    canvas_base = bg.copy()
    neutral_canvas = canvas_base.copy()
    neutral_canvas[y_off:y_off + side, x_off:x_off + side] = cv2.resize(src_rgb, (side, side))

    # ---- frame source + virtual camera ----
    if args.driving_video:
        reader = VideoFileReader(args.driving_video)
        mirror = False
    else:
        reader = WebcamReader(args.camera_index, args.cam_width, args.cam_height, args.cam_fps)
        mirror = not args.no_mirror
    reader.start()
    t0 = time.time()
    while reader.latest() is None:
        if time.time() - t0 > 10:
            sys.exit("frame source produced nothing in 10s")
        time.sleep(0.05)
    print("[init] frame source streaming")

    if args.no_virtual_cam:
        cam = None
    else:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=OW, height=OH, fps=args.cam_fps,
                                  fmt=pyvirtualcam.PixelFormat.RGB)
        print(f"[init] virtual camera: {cam.device} {OW}x{OH}@{args.cam_fps}")

    def emit(frame_rgb):
        if cam is not None:
            cam.send(frame_rgb)
        if args.preview:
            cv2.imshow("Eva TRT", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            return cv2.waitKey(1) & 0xFF == ord("q")
        return False

    calibrated = False
    calib_hist = []
    last_frame = neutral_canvas
    n_done, fps_now = 0, 0.0
    fps_t0, fps_n0 = time.time(), 0
    fps_samples = []

    print("[run] live. Ctrl+C to stop.")
    try:
        while True:
            frame = reader.latest()
            if frame is None:
                time.sleep(0.005)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # while uncalibrated, keep re-anchoring on the current frame
            dri_crop, out_crop, out_org, dri_motion = pipe.run(
                frame, pipe.src_imgs[0], pipe.src_infos[0],
                first_frame=not calibrated)

            lmk = pipe.src_lmk_pre
            if out_crop is None or not lmk_is_sane(lmk, w, h):
                # face lost: hold the last good frame, force redetection
                pipe.src_lmk_pre = None
                if calibrated:
                    print("[run] face lost - holding last pose")
                calibrated = False
                calib_hist = []
                if emit(last_frame):
                    break
                continue

            if not calibrated:
                # calibration gate: anchor only on a frontal, still pose
                x_d = dri_motion[0]
                yaw_deg = float(np.mean(x_d["yaw"]))
                pitch_deg = float(np.mean(x_d["pitch"]))
                center = lmk.mean(0)
                tnow = time.time()
                calib_hist.append((tnow, center, yaw_deg, pitch_deg))
                calib_hist = [c for c in calib_hist if tnow - c[0] <= 0.8]
                centers = np.stack([c[1] for c in calib_hist])
                stable = (len(calib_hist) >= 3
                          and all(abs(c[2]) < 15 and abs(c[3]) < 12 for c in calib_hist)
                          and np.ptp(centers, axis=0).max() < 8)
                if not stable:
                    last_frame = neutral_canvas
                    if emit(neutral_canvas):
                        break
                    continue
                calibrated = True
                print("[run] pose calibrated - driving live")

            face = out_org if use_pasteback else out_crop  # RGB
            canvas = canvas_base.copy()
            canvas[y_off:y_off + side, x_off:x_off + side] = cv2.resize(face, (side, side))
            if args.fps_overlay:
                cv2.putText(canvas, f"{fps_now:.1f} fps", (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            last_frame = canvas
            if emit(canvas):
                break

            n_done += 1
            if args.max_frames and n_done >= args.max_frames:
                break
            if n_done - fps_n0 >= 60:
                tnow = time.time()
                fps_now = (n_done - fps_n0) / (tnow - fps_t0)
                fps_samples.append(fps_now)
                fps_t0, fps_n0 = tnow, n_done
                print(f"[run] {fps_now:.1f} fps | frames {n_done}")
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        if cam is not None:
            cam.close()
        if args.preview:
            cv2.destroyAllWindows()
        if fps_samples:
            print(f"[done] {n_done} frames | median {np.median(fps_samples):.1f} fps")
        else:
            print(f"[done] {n_done} frames")


if __name__ == "__main__":
    main()
