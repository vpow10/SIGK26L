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


def mask_to_np(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"Expected mask [1, H, W], got {tuple(mask.shape)}")
    return mask.detach().cpu().numpy().squeeze(0)


def crop_to_mask_bbox(image_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_np > 0.5)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("Mask is empty")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return image_np[y0:y1, x0:x1]


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


def compute_sne(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff = pred - target
    return float(torch.sum(diff**2).item())


def compute_psnr_hole(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    pred_np = tensor_to_image_np(pred)
    target_np = tensor_to_image_np(target)
    mask_np = mask_to_np(mask)

    pred_crop = crop_to_mask_bbox(pred_np, mask_np)
    target_crop = crop_to_mask_bbox(target_np, mask_np)

    return float(peak_signal_noise_ratio(target_crop, pred_crop, data_range=1.0))


def compute_ssim_hole(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    pred_np = tensor_to_image_np(pred)
    target_np = tensor_to_image_np(target)
    mask_np = mask_to_np(mask)

    pred_crop = crop_to_mask_bbox(pred_np, mask_np)
    target_crop = crop_to_mask_bbox(target_np, mask_np)

    return float(
        structural_similarity(
            target_crop,
            pred_crop,
            channel_axis=2,
            data_range=1.0,
        )
    )


@torch.no_grad()
def compute_lpips_hole(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    mask_np = mask_to_np(mask)
    ys, xs = np.where(mask_np > 0.5)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("Mask is empty")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    pred_crop = pred[:, y0:y1, x0:x1]
    target_crop = target[:, y0:y1, x0:x1]

    model = get_lpips_model()
    pred_batch = pred_crop.unsqueeze(0) * 2.0 - 1.0
    target_batch = target_crop.unsqueeze(0) * 2.0 - 1.0

    value = model(pred_batch, target_batch)
    return float(value.item())


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    return {
        "psnr_full": compute_psnr(pred, target),
        "ssim_full": compute_ssim(pred, target),
        "lpips_full": compute_lpips(pred, target),
        "psnr_hole": compute_psnr_hole(pred, target, mask),
        "ssim_hole": compute_ssim_hole(pred, target, mask),
        "lpips_hole": compute_lpips_hole(pred, target, mask),
        "sne": compute_sne(pred, target),
    }


def average_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        raise ValueError("metric_dicts is empty")

    keys = metric_dicts[0].keys()
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys
    }
