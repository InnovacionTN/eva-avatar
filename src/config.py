import os

# Force the NVIDIA dGPU; never fall back to the Intel iGPU or CPU silently.
# Must be set before torch initializes CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

# onnxruntime-gpu resolves cudart/cublas/cudnn via the DLL search path, not via
# DLLs already loaded by torch — expose torch's bundled CUDA libs to it.
TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
os.add_dll_directory(TORCH_LIB)
os.environ["PATH"] = TORCH_LIB + os.pathsep + os.environ.get("PATH", "")

DEVICE = "cuda:0"
assert torch.cuda.is_available(), "CUDA not available — refusing to run on CPU/iGPU."
_gpu_name = torch.cuda.get_device_name(0)
assert "RTX 2000 Ada" in _gpu_name, f"Unexpected GPU: {_gpu_name}"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET_FPS = 30
SOURCE_IMAGE = os.path.join(PROJECT_ROOT, "assets", "eva_source.png")
SOURCE_IMAGE_RAW = os.path.join(PROJECT_ROOT, "assets", "eva_source_raw.png")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "models")
LIVEPORTRAIT_DIR = os.path.join(PROJECT_ROOT, "third_party", "LivePortrait")

# one-euro filter defaults (tune in Phase 2)
SMOOTH_MIN_CUTOFF = 1.0
SMOOTH_BETA = 0.3
