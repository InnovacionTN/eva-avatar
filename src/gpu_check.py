import json
import os
import platform

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
}, indent=2))
