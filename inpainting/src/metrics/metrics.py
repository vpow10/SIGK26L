from typing import Dict, List

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_model = None


def get_lpips_model() -> lpips.LPIPS:
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex")
        _lpips_model.eval()
    return _lpips_model


def tensor_to_image_np(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W], got {tuple(image.shape)}")

    return image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = tensor_to_image_np(pred)
    target_np = tensor_to_image_np(target)
    return float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = tensor_to_image_np(pred)
    target_np = tensor_to_image_np(target)
    return float(
        structural_similarity(
            target_np,
            pred_np,
            channel_axis=2,
            data_range=1.0,
        )
    )


@torch.no_grad()
def compute_lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    model = get_lpips_model()

    pred_batch = pred.unsqueeze(0) * 2.0 - 1.0
    target_batch = target.unsqueeze(0) * 2.0 - 1.0

    value = model(pred_batch, target_batch)
    return float(value.item())


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "lpips": compute_lpips(pred, target),
    }


def average_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        raise ValueError("metric_dicts is empty")

    keys = metric_dicts[0].keys()
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys
    }
