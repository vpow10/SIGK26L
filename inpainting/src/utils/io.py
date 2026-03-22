from pathlib import Path

import numpy as np
import torch
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_tensor_image(image: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255.0).round().astype(np.uint8)
    Image.fromarray(image_np).save(output_path)
