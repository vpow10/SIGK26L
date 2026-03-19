from typing import Dict, Tuple

import numpy as np
import torch


def generate_square_mask(
    height: int,
    width: int,
    patch_size: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Generate a binary square mask of shape [1, H, W].

    Mask convention:
        1.0 -> missing region
        0.0 -> known region
    """
    if patch_size > height or patch_size > width:
        raise ValueError(
            f"Patch size {patch_size} does not fit in image of size ({height}, {width})"
        )

    top = int(rng.integers(0, height - patch_size + 1))
    left = int(rng.integers(0, width - patch_size + 1))

    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[:, top : top + patch_size, left : left + patch_size] = 1.0

    metadata = {
        "top": top,
        "left": left,
        "patch_size": patch_size,
    }
    return mask, metadata


def apply_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply binary mask to an image tensor.

    Args:
        image: Tensor [3, H, W] in [0, 1]
        mask: Tensor [1, H, W], with 1 meaning missing region

    Returns:
        Masked image where missing region is zeroed out.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected image shape [C, H, W], got {tuple(image.shape)}")
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape [1, H, W], got {tuple(mask.shape)}")
    if image.shape[1:] != mask.shape[1:]:
        raise ValueError(
            f"Image and mask spatial shapes do not match: "
            f"{tuple(image.shape)} vs {tuple(mask.shape)}"
        )

    return image * (1.0 - mask)


def mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    """
    Convert mask [1, H, W] float tensor to uint8 [H, W] for OpenCV.
    """
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"Expected mask shape [1, H, W], got {tuple(mask.shape)}")

    mask_np = mask.squeeze(0).cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    return mask_np
