# Eva — Photorealistic Motion-Driven Avatar (Implementation Plan v2)

> **For Claude Code.** Execute phase by phase. Each phase has a **Goal**, **Steps**, and a **Checkpoint** that must pass before continuing. Stop and report at every checkpoint. Do not skip Phase 0.
>
> **This plan is tuned to a known target machine** (see §1). If `nvidia-smi` in Phase 0 does NOT match, stop and report before proceeding.

---

## 0. Decision summary (read first)

**What we are building:** a live, photorealistic avatar of "Eva" (the source portrait) whose face, head, and expressions are driven in real time by the operator's webcam, with output piped to a **virtual camera** so it appears as a selectable camera source in Meet / Teams / Zoom / OBS.

**Core approach (DEFAULT — "presenter" mode):** animate the source portrait with **LivePortrait** (`KwaiVGI/LivePortrait`). The operator is invisible; only Eva is shown. The body/shirt/background in the photo stay static; head, face, eyes, and mouth move.

**Alternative ("face-swap" mode):** put Eva's *face* onto the operator's live webcam body with **Deep-Live-Cam**. Real body motion, but the operator's own hair/body/background show. Swap-in described in **Appendix A**.

> **Claude Code: confirm with the user which mode is active before Phase 1. Default to presenter mode.**

**Source asset:** the portrait `Eva_nuevo_logo.png`. Copy it into the repo as `assets/eva_source_raw.png` and never overwrite it. The prepped version is produced in Phase 1.

**Asset note:** Eva is a synthetic brand character (Tiendas Neto presenter), the operator's own asset — not a real identifiable person being impersonated.

---

## 1. Target machine (KNOWN — confirm, don't re-survey)

| Component | Value |
|---|---|
| GPU (use this one) | **NVIDIA RTX 2000 Ada Generation Laptop GPU**, **8 GB VRAM** |
| GPU driver / CUDA | Driver **565.126**, **CUDA 12.5** runtime supported |
| Second GPU (ignore) | Intel Arc Pro iGPU — **must not** be used for inference |
| CPU | Intel Core Ultra 9 185H (16C / 22T) |
| System Python | **3.13.7 — DO NOT USE for this project** |
| OS | Windows (corporate; possible Group Policy restrictions) |

**Feasibility: real-time mode is GO.** 8 GB VRAM clears LivePortrait's ~6 GB need with headroom; expect **~20–30 fps** live. No cloud/offline fallback required.

**Two hard rules derived from this machine:**
1. **Use Python 3.10 in a dedicated environment.** The system 3.13.7 will break the PyTorch / InsightFace / onnxruntime stack (missing wheels, native build failures). See Phase 0.
2. **Pin all inference to the NVIDIA dGPU (CUDA device 0).** With the Intel Arc iGPU also present, guard against silent CPU/iGPU fallback. See `config.py` in Phase 0 and the guard in Phase 2.

---

## 2. Target repository structure

```
eva-avatar/
├── README.md
├── requirements.txt
├── environment.yml              # conda env (python 3.10) — preferred path
├── .gitignore
├── assets/
│   ├── eva_source_raw.png        # untouched copy of the upload
│   └── eva_source.png            # prepped: square, face-centered, 512x512
├── models/                       # LivePortrait weights (gitignored)
├── src/
│   ├── config.py                 # paths, fps target, smoothing, DEVICE pinning
│   ├── gpu_check.py              # Phase 0 feasibility + device-pin verification
│   ├── prep_source.py            # crop/align/validate the source image
│   ├── run_offline.py            # source image + driving video -> output video
│   ├── run_realtime.py           # webcam -> animated Eva -> virtual camera
│   └── smoothing.py              # one-euro filter for jitter reduction
├── third_party/
│   └── LivePortrait/             # cloned upstream
└── scripts/
    ├── setup_env.ps1
    └── download_weights.ps1
```

`config.py` device pin (create early, import everywhere):

```python
import os, torch
# Force the NVIDIA dGPU; never fall back to the Intel iGPU or CPU silently.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEVICE = "cuda:0"
assert torch.cuda.is_available(), "CUDA not available — refusing to run on CPU/iGPU."
assert "RTX 2000 Ada" in torch.cuda.get_device_name(0) or torch.cuda.get_device_name(0), \
    f"Unexpected GPU: {torch.cuda.get_device_name(0)}"
TARGET_FPS = 30
SOURCE_IMAGE = "assets/eva_source.png"
# one-euro filter defaults (tune in Phase 2)
SMOOTH_MIN_CUTOFF = 1.0
SMOOTH_BETA = 0.3
```

---

## Phase 0 — Environment & device gate (the part that goes wrong if rushed)

**Goal:** a Python **3.10** environment with CUDA-enabled PyTorch that runs on the **RTX 2000 Ada**, verified before any app code.

**Steps**

1. **Verify the machine** with `nvidia-smi`. Confirm `RTX 2000 Ada`, ~8188 MiB VRAM, CUDA 12.x. If it differs, **stop and report**.

2. **Get a Python 3.10 environment.** Preferred = conda (cleanest for this ML stack and side-steps the system 3.13). If conda/Miniforge is available or installable in user scope:
   ```powershell
   conda create -n eva python=3.10 -y
   conda activate eva
   ```
   If conda is blocked by Group Policy, install Python 3.10 from the python.org **user-scope** installer and create a venv:
   ```powershell
   py -3.10 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   **Verify:** `python --version` → `Python 3.10.x`. If it prints 3.13, the wrong interpreter is active — fix before continuing.

3. **Install CUDA-enabled PyTorch.** The 565.x driver supports the CUDA 12.1 runtime; use the `cu121` wheels (most compatible with LivePortrait's tested torch 2.3):
   ```powershell
   pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Clone LivePortrait and install its deps:**
   ```powershell
   git clone https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
   pip install -r third_party/LivePortrait/requirements.txt
   ```
   (Includes `onnxruntime-gpu`, `insightface`, etc. If `insightface` fails to build, install the prebuilt wheel for cp310/win first.)

5. **Write `src/gpu_check.py`** that prints a JSON summary and asserts the device pin:
   ```python
   import torch, json
   print(json.dumps({
     "python": __import__("platform").python_version(),
     "torch": torch.__version__,
     "cuda_available": torch.cuda.is_available(),
     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
     "vram_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0,
   }, indent=2))
   ```

**Checkpoint 0**
- `python --version` → 3.10.x (NOT 3.13).
- `python src/gpu_check.py` → `cuda_available: true`, device contains "RTX 2000 Ada", `vram_gb` ≈ 8.
- Report the JSON and the confirmed mode (presenter vs face-swap) to the user before Phase 1.

---

## Phase 1 — Offline reenactment (prove the core before webcams)

**Goal:** `eva_source.png` + a short driving video → an output video where Eva mimics the driver. Validates the model in isolation.

**Steps**
1. `scripts/download_weights.ps1` — fetch LivePortrait pretrained weights into `models/`. Verify completeness; fail loudly on partial downloads.
2. `src/prep_source.py`:
   - Load `assets/eva_source_raw.png`, detect the face, crop to a centered square, resize to **512×512**, save `assets/eva_source.png`.
   - Validate: exactly one face, front-facing, eyes open. On failure, report and ask for a better source frame (see Appendix C checklist).
3. Record a short driving clip: 5–10 s of the operator turning their head and talking.
4. `src/run_offline.py`: wrap LivePortrait inference → `out/eva_offline.mp4`. Expose flags for stitching and eye/lip retargeting. Use `DEVICE` from `config.py`.

**Checkpoint 1**
- `out/eva_offline.mp4` exists; Eva's head turns and mouth moves with the driver; the Neto shirt and background stay coherent (no warping/melting).
- Print achieved processing fps. Report to user with the sample path.

---

## Phase 2 — Real-time webcam → virtual camera

**Goal:** drive Eva live from the webcam to a virtual camera at ≥ 15 fps (target 20–30).

**Steps**
1. **Virtual-camera backend:** install/start **OBS Virtual Camera** (recommended) or `pip install pyvirtualcam` with the OBS DLL backend.
2. `src/run_realtime.py` main loop:
   - Open webcam via OpenCV (1280×720 @ 30 fps; drop to 640×480 if fps suffers).
   - **Cache the source encoding ONCE at startup** — re-encoding Eva per frame destroys fps.
   - Per frame: detect/track driver face → compute motion params → LivePortrait inference on `cuda:0` against the cached source → animated Eva frame → write to virtual camera.
   - Run inference on a **separate thread** from capture/output so a slow frame doesn't stall the webcam read.
3. `src/smoothing.py`: **one-euro filter** on head pose + expression params to kill jitter. Params from `config.py` (`SMOOTH_MIN_CUTOFF`, `SMOOTH_BETA`).
4. Robustness: on face-detection miss, **hold the last good pose** (never snap to neutral). Add `--mirror` and a toggleable FPS overlay.
5. **GPU-pin guard:** at startup, assert `torch.cuda.get_device_name(0)` is the RTX and log VRAM use after warmup; abort with a clear message if it landed on CPU/iGPU.

**Checkpoint 2**
- "Eva" is selectable as a camera in OBS / a Meet test page.
- Sustained ≥ 15 fps (aim 20–30); head + mouth + blink track the operator with < ~150 ms perceived latency.
- Motion smooth (no visible jitter) with the filter on.

---

## Phase 3 — Quality & robustness pass

**Goal:** presentable, not just functional.

**Steps**
1. Tune retargeting so Eva's neutral matches the source (no permanent half-smile / wandering gaze).
2. Colour/lighting: match output tone to the source; optional light denoise on output.
3. Edge cases: operator leaves frame; multiple faces (lock to largest/centered); webcam disconnect/reconnect.
4. Performance: profile the loop; keep model load, source encoding, and warmup out of the hot path; confirm VRAM stays under 8 GB.
5. Make everything config/CLI driven (fps, resolution, smoothing, source path).

**Checkpoint 3**
- Runs 10+ minutes with no crash, fps drift, or VRAM growth.
- Recovers cleanly from the operator stepping out and back into frame.

---

## Phase 4 — Packaging & handoff

**Goal:** a non-technical operator can launch Eva.

**Steps**
1. One launcher: `setup_env.ps1` then a `run` entry point; ensure OBS Virtual Camera is running or instruct the user.
2. `README.md`: prerequisites, the Python-3.10 rule, how to start the virtual cam, how to select "Eva" in Meet/Teams, troubleshooting (low fps, face-not-detected, wrong-GPU).
3. Optional (only if requested): a minimal local control UI (Tkinter or localhost page) to start/stop, switch source image, toggle mirror, adjust smoothing.

**Checkpoint 4**
- A clean clone + setup + run produces a working virtual camera per the README, no undocumented manual steps.

---

## 3. Risks & handling (this machine)

- **Python 3.13 contamination** → biggest practical risk. If the wrong interpreter activates, builds fail cryptically. Phase 0 hard-checks `python --version`.
- **Inference lands on iGPU/CPU** → silent slowdown. `config.py` pins `CUDA_VISIBLE_DEVICES=0` + Phase 2 startup assert.
- **VRAM ceiling (8 GB)** → comfortable but not infinite. Keep batch=1, cache source once, watch VRAM in Phase 3.
- **Corporate GPO blocks installs / virtual cam driver** → if conda/pip/OBS install is blocked, surface it immediately; fall back to the user-scope python.org installer, and if the virtual-cam driver can't be installed, document offline render (Phase 1) as the deliverable instead.
- **Uncanny valley** → photorealistic + imperfect motion can read worse than stylized. Set expectations: great for a branded presenter on a call, not flawless.

---

## 4. Acceptance criteria (definition of done)

- [ ] Phase 0: Python 3.10 env, `cuda_available: true` on the RTX 2000 Ada, mode confirmed.
- [ ] Phase 1: clean offline Eva-mimic video (Checkpoint 1).
- [ ] Phase 2: live webcam → virtual camera at ≥ 15 fps, smoothed (Checkpoint 2).
- [ ] Phase 3: stable 10+ min, recovers from face loss, VRAM under 8 GB (Checkpoint 3).
- [ ] Phase 4: one-command launch + README (Checkpoint 4).

---

## Appendix A — Face-swap mode (Deep-Live-Cam alternative)

If the user wants Eva's *face on their live body* instead of an animated portrait:
- Replace Phases 1–2 with `hacksider/Deep-Live-Cam`: real-time face swap from a single source face onto the live webcam, output to a virtual camera.
- `eva_source.png` becomes the swap source face (same Appendix C prep applies).
- Same Python-3.10 + CUDA + GPU-pin rules; same Phases 0 / 3 / 4.
- Trade-off: more lifelike body/hand motion, but the operator's own hair/body/background show, not Eva's.

## Appendix B — (Not needed on this machine) cloud / offline fallback

Retained only as a contingency if the GPU becomes unavailable or GPO blocks the virtual-cam driver: run Phase 1 offline rendering (no real-time, no virtual cam) locally or on a rented T4/A10-class GPU instance.

## Appendix C — Source-image prep checklist

- Square crop, face centered, ~512×512.
- Single front-facing face, eyes open, neutral-to-slight expression.
- Even lighting, no heavy shadow across the face.
- Keep the Neto shirt + clean background in frame (static but visible).
- Keep `eva_source_raw.png` as an untouched master; never overwrite the original upload.

---

## Quick-start command block (Phase 0, copy-paste)

```powershell
# 1. environment (conda preferred)
conda create -n eva python=3.10 -y
conda activate eva
python --version            # must print 3.10.x

# 2. CUDA PyTorch (cu121 wheels, compatible with the 565 driver)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 3. LivePortrait
git clone https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
pip install -r third_party/LivePortrait/requirements.txt

# 4. verify GPU
python src/gpu_check.py     # expect cuda_available: true, RTX 2000 Ada, ~8 GB
```
