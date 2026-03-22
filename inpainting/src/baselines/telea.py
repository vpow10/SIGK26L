from typing import Union

import cv2
import numpy as np
import torch
from src.utils.masks import mask_to_uint8


def tensor_to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3, H, W], got {tuple(image.shape)}")

    image_np = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image_np = (image_np * 255.0).round().astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr


def bgr_uint8_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape [H, W, 3], got {tuple(image.shape)}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
    return tensor


def telea_inpaint(
    masked_image: torch.Tensor,
    mask: torch.Tensor,
    inpaint_radius: float = 3.0,
) -> torch.Tensor:
    image_bgr = tensor_to_bgr_uint8(masked_image)
    mask_uint8 = mask_to_uint8(mask)

    output_bgr = cv2.inpaint(
        image_bgr,
        mask_uint8,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    output_tensor = bgr_uint8_to_tensor(output_bgr)
    return output_tensor
