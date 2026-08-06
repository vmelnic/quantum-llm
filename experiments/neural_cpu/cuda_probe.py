"""Machine-readable CUDA PyTorch environment probe."""

from __future__ import annotations

import json

import torch


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

print(
    json.dumps(
        {
            "status": "ready",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
        }
    )
)
