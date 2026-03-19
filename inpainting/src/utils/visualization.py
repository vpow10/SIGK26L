import numpy as np
import torch


def tensor_to_image_np(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got {tuple(tensor.shape)}")

    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    image = tensor.permute(1, 2, 0).numpy()
    return image


def mask_to_image_np(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"Expected [1, H, W], got {tuple(mask.shape)}")

    return mask.squeeze(0).detach().cpu().numpy()
