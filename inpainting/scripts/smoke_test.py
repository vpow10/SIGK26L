import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


def main() -> None:
    config = load_config("configs/base.yaml")
    set_seed(config["seed"])

    requested_device = config["train"]["device"]
    device = get_device(requested_device)

    print("=== Smoke Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Requested device: {requested_device}")
    print(f"Resolved device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Running on CPU.")

    x = torch.randn(2, 3).to(device)
    y = torch.randn(2, 3).to(device)
    z = x + y

    print("Tensor operation successful.")
    print(f"Output shape: {z.shape}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
