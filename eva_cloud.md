# Eva in the Cloud — Deployment & Retrofit Plan

> **Audience:** an AI coding agent (Claude Code) running *inside the target GCP
> VM*, plus the human operator. Read this top to bottom before touching code.
> The repo is the existing **Eva talks** project (live conversational Spanish
> avatar: mic → Gemini Live → speakers, with the response audio driving Eva's
> lips on FasterLivePortrait TensorRT models).
>
> **The one thing that changes in the cloud:** a GCP VM is headless — no mic,
> no speakers, no webcam, no monitor. So the *local* I/O (`sounddevice`,
> OpenCV `imshow` preview window, `pyvirtualcam`, `cv2.VideoCapture` webcam)
> **cannot be used**. We replace all of it with **WebRTC**: the user's browser
> becomes the mic, camera, speakers, and "window." The GPU just renders Eva at
> 60 fps and streams her out.

---

## 0. Goal & success criteria

1. Eva renders at a **steady 60 fps** on the VM GPU (frame budget < 16.7 ms).
2. A user opens a **URL in their laptop browser**, grants mic+camera, and has a
   real-time spoken Spanish conversation with Eva — they hear her voice, see her
   lip-synced face, and she can see them (vision).
3. Glass-to-glass latency feels conversational (target < ~400 ms end-to-end;
   most of that is Gemini + network, not render).

If you only get the avatar rendering into the void (no browser session), you are
**not done** — that was already true locally.

---

## 1. Target VM characteristics

Create (or verify you are on) this machine. **Recommended: NVIDIA L4.**

| Item | Value | Notes |
|---|---|---|
| Machine type | `g2-standard-8` | 1× **NVIDIA L4 24 GB**, 8 vCPU, 32 GB RAM |
| GPU arch | Ada Lovelace (`sm_89`) | Same family as the dev box's RTX 2000 Ada → TRT plugins/ops match, but **engines must be rebuilt** (see §5) |
| OS image | **Ubuntu 22.04 LTS** (`ubuntu-2204-lts`) | x86_64. Avoid the "Deep Learning VM" unless you accept its pinned CUDA. |
| Boot disk | 100 GB SSD (`pd-ssd`) | Models + engines + CUDA/TRT ≈ 30–50 GB; leave headroom |
| Driver | NVIDIA **R550+** (CUDA 12.4-capable) | Must satisfy torch cu121/cu124 + TRT 10 |
| Region | any with L4 stock (`us-central1`, `us-east4`, `europe-west4`, …) | |
| Firewall | allow **tcp:443** (or your chosen TLS port) + **udp:3478, udp:49152-65535** | WebRTC: HTTPS signaling + STUN + media (SRTP/ICE). See §8.3. |
| **No-risk alt** | `a2-highgpu-1g` (A100 40 GB) | ~4× the cost; only if L4 measurably misses 60 fps after §9 tuning |

**Cost reminder (us-central1, on-demand):** L4 `g2-standard-8` ≈ **$0.85/hr**
(~$620/mo if left on 24/7) + ~$8.50/mo for the 100 GB disk (charged even when
stopped). A100 ≈ **$3.67/hr**. **Stop the VM when not in use.** Do **not** use
Spot/preemptible for live sessions (mid-conversation preemption); Spot is fine
for offline engine builds/benchmarks.

### Create it (from the operator's workstation, if not already provisioned)

```bash
gcloud compute instances create eva-vm \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-ssd \
  --metadata=install-nvidia-driver=True

# open the WebRTC ports
gcloud compute firewall-rules create eva-webrtc \
  --allow=tcp:443,udp:3478,udp:49152-65535 \
  --target-tags=eva --direction=INGRESS
gcloud compute instances add-tags eva-vm --tags=eva --zone=us-central1-a
```

---

## 2. Current architecture (local / Windows) — what exists today

All in `src/eva_platica.py` (~909 lines). Key components:

| Component | Class / fn | Local device it touches | Cloud fate |
|---|---|---|---|
| Speaker out + audio clock | `SpeakerStream` | `sounddevice.OutputStream` | **Rewire** → feed an aiortc audio track; keep the envelope/clock logic |
| Mic in | `MicStream` | `sounddevice.InputStream` | **Replace** → frames come from the browser's WebRTC audio track |
| User VAD | `UserVoiceMeter` | (pure math) | **Keep** |
| Phoneme mouth | `PhonemeScheduler`, `g2p_es` | (pure math) | **Keep** |
| Gemini Live session | `GeminiVoice` | network + the above | **Keep**, swap its mic/vision sources |
| Vision (webcam→Gemini) | `vision_sender()` | `cv2.VideoCapture(CAP_DSHOW)` | **Replace** → JPEG frames from the browser's camera track |
| Avatar render (GPU) | `EvaRenderer.frame()` | TensorRT (FLP) | **Keep** (this is the 60 fps target) |
| Output compose + display | `OutputSink` | `cv2.imshow` + `pyvirtualcam` | **Replace** → push frames to an aiortc video track |
| Entrypoint | `main()` | argparse loop | **Refactor** into the server (see §7) |

**Platform note:** the local build lives under
`third_party/FLP-win/FasterLivePortrait-windows/` (Windows binaries) and uses
Windows-only paths (`cv2.CAP_DSHOW`, OBS virtual cam). On Linux use the
cross-platform source already vendored at **`third_party/FasterLivePortrait/`**
and build engines there (§5). The WebRTC retrofit conveniently *deletes* most of
the platform-specific code.

---

## 3. Target architecture (cloud / Linux + WebRTC)

```
  ┌─────────────── user's laptop browser ───────────────┐
  │  getUserMedia(mic+cam)  ──Opus/VP8──▶                │
  │  <video> plays Eva     ◀──Opus/VP8── RTCPeerConn     │
  └───────────────▲───────────────────────┬─────────────┘
                  │ HTTPS (SDP offer/answer)│ media (SRTP/ICE/STUN)
  ┌───────────────┴─────────────────────────▼────────────── GCP L4 VM ──┐
  │  aiohttp signaling + static index.html                               │
  │  aiortc RTCPeerConnection                                            │
  │    • inbound audio track  → resample 48k→16k → mic_q → Gemini        │
  │    • inbound video track  → ~1 fps JPEG       → Gemini (vision)      │
  │    • outbound audio track ← Gemini 24k PCM (resample→48k Opus)       │
  │    • outbound video track ← EvaRenderer.frame() RGB → VideoFrame     │
  │  GeminiVoice (Live API)  ←─ GEMINI_API_KEY                           │
  │  EvaRenderer (TensorRT, L4)  ── 60 fps render loop                   │
  └──────────────────────────────────────────────────────────────────────┘
```

The Gemini Live audio path is **unchanged** (still 16 kHz up / 24 kHz down PCM).
Only the *endpoints* of audio/video move from local devices to WebRTC tracks.

---

## 4. VM setup — step by step

Run as the agent on the VM. Stop at the first command that fails and diagnose.

### 4.1 System + NVIDIA driver

```bash
sudo apt-get update && sudo apt-get install -y build-essential git wget \
     python3.10 python3.10-venv python3.10-dev ffmpeg libsm6 libxext6 \
     libopus0 libvpx7 pkg-config

# Driver (skip if the create-time metadata already installed R550+):
nvidia-smi   # must print an L4 and CUDA >= 12.4. If not:
#   sudo apt-get install -y nvidia-driver-550 && sudo reboot
```

`nvidia-smi` **must** show `NVIDIA L4`. If it shows nothing, fix the driver
before going further — nothing downstream works without it.

### 4.2 Get the repo

```bash
cd ~ && git clone <THIS_REPO_URL> eva && cd eva
# OR: the operator scp's the project tree here. Ensure these exist:
#   src/eva_platica.py, third_party/FasterLivePortrait/, assets/eva_source.png,
#   models/ (or download — see §5), .env (GEMINI_API_KEY=...)
```

### 4.3 Python env

```bash
python3.10 -m venv ~/eva/venv
source ~/eva/venv/bin/activate
pip install --upgrade pip wheel

# 1) FasterLivePortrait base deps (cross-platform list):
pip install -r third_party/FasterLivePortrait/requirements.txt

# 2) CUDA-matched torch + onnxruntime for the L4 driver (CUDA 12.x):
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.18.0

# 3) TensorRT for Linux (match the driver; TRT 10.x for R550):
pip install tensorrt==10.2.0 pycuda

# 4) Eva cloud additions (Gemini + WebRTC):
pip install -r requirements_cloud.txt
```

> If `tensorrt` pip wheels fight the driver, install TRT from the NVIDIA apt repo
> or tarball matched to the CUDA version `nvidia-smi` reports. The hard
> constraint: **torch CUDA build, onnxruntime-gpu, and tensorrt must all target
> the same CUDA major.minor** as the installed driver.

### 4.4 Secrets

```bash
# .env at repo root (NEVER commit it; it is gitignored):
echo "GEMINI_API_KEY=<key>" > ~/eva/.env
```

---

## 5. Rebuild the TensorRT engines for the L4

The local `.trt` files were built for the **RTX 2000 Ada (Windows, TRT 9)** and
**will not load** on the L4 (Linux, TRT 10). Rebuild from the ONNX sources.

```bash
cd ~/eva/third_party/FasterLivePortrait

# 1) Get ONNX weights (HuggingFace: warmshao/FasterLivePortrait):
huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints

# 2) Convert ONNX → TRT for THIS GPU. The repo ships a converter script;
#    on Linux it is typically:
sh scripts/all_onnx2trt.sh
#    (if absent, convert each model with trtexec, FP16, e.g.:
#     trtexec --onnx=checkpoints/liveportrait_onnx/warping_spade-fix.onnx \
#             --saveEngine=checkpoints/liveportrait_onnx/warping_spade-fix.trt \
#             --fp16 )
```

Engines land in `checkpoints/liveportrait_onnx/*.trt` — the same relative paths
`configs/trt_infer.yaml` expects. Eva's loader (`EvaRenderer`) `chdir`s to the
FLP root, so keep that directory layout. The models Eva loads:
`warping_spade-fix`, `motion_extractor`, `landmark`, `face_analysis`
(`retinaface_det` + `face_2dpose_106`), `appearance_feature_extractor`,
`stitching`, `stitching_eye`, `stitching_lip`.

> Eva's config currently points at `third_party/FLP-win/.../configs/trt_infer.yaml`
> (see `eva_platica.py` top, the `CANDIDATES` glob). On Linux, point it at
> `third_party/FasterLivePortrait/configs/trt_infer.yaml` instead (update the
> glob or symlink the dir).

---

## 6. Dependency manifest

Two files, installed into the **same venv** in this order:

1. `third_party/FasterLivePortrait/requirements.txt` — FLP base
   (`omegaconf, onnx, pycuda, numpy, opencv-python, scikit-image, insightface,
   mediapipe, huggingface_hub[cli], torchgeometry, soundfile, ffmpeg-python`).
   On a headless VM, prefer **`opencv-python-headless`** over `opencv-python`.
2. `requirements_cloud.txt` (in repo root, created alongside this plan) — Gemini
   + WebRTC additions, and the list of packages to **remove**.

**Removed vs local:** `sounddevice`, `pyvirtualcam` (and the webcam use of
`cv2.VideoCapture`). They touch hardware the VM doesn't have.

---

## 7. The WebRTC retrofit — what to build

Create **`src/eva_server.py`** (new). It reuses Eva's brains and replaces only
the I/O ends. Do **not** rewrite `EvaRenderer`, `PhonemeScheduler`, `g2p_es`,
`UserVoiceMeter`, or the `GeminiVoice` session logic — import and reuse them.

### 7.1 Audio out (Eva's voice) — adapt `SpeakerStream`

`SpeakerStream` today plays 24 kHz PCM via `sounddevice` **and** exposes the
audio clock (`now_ms`, `fed_ms`, `level`, `speaking`) the phoneme scheduler
aligns to. Keep all of that; replace only the device sink:

- Delete the `sd.OutputStream`. Keep `_buf`, `_level`, `_played`, `_fed`.
- Add a `pull(n_samples)` the aiortc audio track calls each 20 ms; it drains
  `_buf`, advances `_played`, updates `_level` — i.e. move the old `_callback`
  body into `pull`. The audio clock keeps working unchanged.
- Build an `aiortc.mediastreams.MediaStreamTrack` (`kind="audio"`) whose
  `recv()` returns a 48 kHz `av.AudioFrame`: pull 24 kHz PCM from the buffer,
  resample 24k→48k (PyAV `av.AudioResampler`), return Opus-encodable frames.

### 7.2 Audio in (user mic) — replace `MicStream`

- The inbound browser audio track yields 48 kHz `av.AudioFrame`s.
- Resample 48k→16k PCM16 mono, feed `UserVoiceMeter`, and `put` chunks on the
  same `mic_q` that `GeminiVoice._session_loop.sender()` already drains.
- Keep the **gate**: mute the uplink while `speaker.speaking()` (no barge-in on
  an open channel) unless `--allow-barge-in`. WebRTC has browser-side echo
  cancellation, so you *may* relax the gate — test before enabling barge-in.

### 7.3 Vision in (user camera) — replace `vision_sender()`

- Pull frames from the inbound browser **video** track (av.VideoFrame → ndarray).
- Throttle to ~1 fps, JPEG-encode (`cv2.imencode`, quality 70, max width 768),
  `session.send_realtime_input(video=Blob(... image/jpeg))`. Same as today, just
  a different frame source. No `CAP_DSHOW`, no `VideoCapture`.

### 7.4 Video out (Eva's face) — replace `OutputSink`

- Run the existing render loop (`EvaRenderer.frame(level, speaking, mouth_delta,
  listening)`) at the target fps in a thread; push each RGB frame (composited
  onto the blurred bg canvas as `OutputSink` does today) into a 1-slot queue.
- An `aiortc` video `MediaStreamTrack.recv()` wraps the latest frame as an
  `av.VideoFrame` (`format="rgb24"`), sets pts/time_base for the fps, returns it.
- Drop `cv2.imshow` / `cv2.waitKey` / `pyvirtualcam` entirely.

### 7.5 Signaling + page (`aiohttp`)

- `GET /` → static `index.html` (getUserMedia, RTCPeerConnection, attaches Eva's
  tracks to a `<video autoplay>`).
- `POST /offer` → take browser SDP offer, add Eva's audio+video tracks, wire the
  inbound tracks to §7.2/§7.3, return SDP answer.
- Serve over **HTTPS** (browsers require a secure context for getUserMedia).
  Use a real cert (Caddy/nginx in front, or a Let's Encrypt cert on the VM's
  external IP/DNS). A self-signed cert works for testing with a browser override.

### 7.6 Reuse `--check`

`run_check()` (connect to Gemini, get one spoken reply, save WAV, exit) needs no
devices — keep it as a fast cloud connectivity test before bringing up WebRTC.

---

## 8. Run & connect

### 8.1 Start the server

```bash
cd ~/eva && source venv/bin/activate
python src/eva_server.py --fps 60 --voice Leda          # default model = gemini-3.1-flash-live-preview
# flags carried over: --voice, --vision (now means "accept browser camera"),
# --allow-barge-in, --lip-gain, --mouth, --out-width/--out-height, --check
```

### 8.2 Connect

Open `https://<VM_EXTERNAL_IP>/` (or your DNS name) on the laptop, allow mic +
camera, and **habla con Eva**. The `<video>` element is "the window."

### 8.3 Networking notes

- Open **tcp:443** (signaling/HTTPS) and **udp:3478 + udp:49152-65535** (STUN +
  media) in the GCP firewall (done in §1).
- Add a public STUN server in the JS (`stun:stun.l.google.com:19302`). For
  picky/symmetric NATs you may need a **TURN** server; the VM has a public IP so
  host-side STUN usually suffices.

---

## 9. Hit 60 fps — verify & tune

1. **Measure raw render cost first** (no Gemini, no WebRTC):
   `python src/bench_stages_trt.py` (already in `src/`) — confirms per-frame
   `warping_spade` + `stitching` time on the L4. Need **< 16.7 ms**; aim < 12 ms
   for headroom. On a 30 W RTX 2000 Ada laptop this was ~60 ms; the L4 (~3.5–4×,
   not power-throttled) should land ~15–18 ms, then tune down.
2. **Confirm engines are FP16** (rebuild with `--fp16`; try **FP8** on Ada/L4 for
   `warping_spade` if still short — biggest single cost).
3. **Pin the GPU clocks**: `sudo nvidia-smi -lgc <max>` and persistence mode
   `sudo nvidia-smi -pm 1` so the L4 doesn't downclock between frames.
4. **Run the render loop on its own thread** (already the design) so WebRTC
   encode/network never stalls the GPU.
5. Watch the `[run] N fps | frames M` log line (already printed) — it should sit
   at ~60. If it caps at the `--fps` value with GPU headroom to spare, you're
   render-bound-free and the cap is the limit; raise `--fps`.
6. If still short after FP8 + clocks → fall back to **A100** (`a2-highgpu-1g`).
   Do this only with bench numbers proving the L4 misses; don't pay 4× on a hunch.

> Note: "60 fps rendered" ≠ "60 fps perceived." WebRTC/Opus/VP8 + network add
> latency and the browser caps at its own refresh. The render target is about
> smoothness; conversational *feel* is dominated by Gemini round-trip + network.

---

## 10. Gotchas / checklist

- [ ] `nvidia-smi` shows **L4** and CUDA ≥ 12.4 before anything else.
- [ ] torch (cu121/cu124) + onnxruntime-gpu + tensorrt all on the **same CUDA**.
- [ ] **Rebuilt** `.trt` engines on the L4 (old Windows/Ada-laptop engines fail).
- [ ] Config glob points at `third_party/FasterLivePortrait/.../trt_infer.yaml`,
      not the Windows `FLP-win` path.
- [ ] `opencv-python-headless` (not `opencv-python`) — no display libs on VM.
- [ ] **No** `sounddevice` / `pyvirtualcam` installed or imported.
- [ ] `.env` has `GEMINI_API_KEY`; it's the **Developer API** key (not Vertex).
- [ ] HTTPS in front of the signaling server (getUserMedia needs secure context).
- [ ] Firewall: tcp:443 + udp:3478 + udp:49152-65535 open.
- [ ] `--check` succeeds (Gemini reachable) before debugging WebRTC.
- [ ] **Stop the VM** when done (`gcloud compute instances stop eva-vm`).

---

## 11. Build order (suggested for the agent)

1. §4 system + driver + venv; `nvidia-smi` green.
2. §5 download ONNX, rebuild TRT engines, fix the config path.
3. Smoke-test the renderer headless: adapt `--mute --max-frames 120` path to
   render into memory (no window) and confirm ~60 fps via the bench.
4. `python src/eva_server.py --check` — Gemini connectivity.
5. §7 build `eva_server.py`: video-out track first (see Eva move in the browser
   with `--mute` synthetic syllables), then audio-out, then mic-in, then vision.
6. §9 tune to 60 fps.
7. Hand the operator the `https://<IP>/` URL.

---

### Reference: proven local versions (dev box, Windows)

`google-genai 2.8.0` · `python-dotenv 1.0.1` · `numpy 1.26.4` ·
`opencv-python 4.10.0.84` · `omegaconf 2.3.0` · `torch 2.3.0+cu121` ·
`tensorrt 9.0.1` (Win) · `onnxruntime-gpu 1.17.0`. Default model
`gemini-3.1-flash-live-preview`, voice `Leda`, mouth driver `fonemas`.
On the L4 bump torch→cu121 2.3.1, TRT→10.x, onnxruntime-gpu→1.18.0 (§4.3).
