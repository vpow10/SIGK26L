from typing import Dict

import numpy as np
import torch
from src.metrics.metrics import compute_all_metrics
from src.utils.reconstruction import blend_prediction_with_known_region
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_losses = []
    hole_losses = []
    valid_losses = []
    perceptual_losses = []

    for batch in tqdm(loader, desc="Train", leave=False):
        model_input = batch["input"].to(device)
        masked_rgb = batch["masked"].to(device)
        mask = batch["mask"].to(device)
        gt = batch["gt"].to(device)

        optimizer.zero_grad()

        pred_rgb = model(model_input)

        loss, stats = criterion(pred_rgb, gt, mask)

        loss.backward()
        optimizer.step()

        total_losses.append(stats["loss_total"])
        hole_losses.append(stats["loss_hole"])
        valid_losses.append(stats["loss_valid"])
        perceptual_losses.append(stats.get("loss_perceptual", 0.0))

    return {
        "loss_total": float(np.mean(total_losses)),
        "loss_hole": float(np.mean(hole_losses)),
        "loss_valid": float(np.mean(valid_losses)),
        "loss_perceptual": float(np.mean(perceptual_losses)),
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    max_metric_samples: int | None = None,
) -> Dict[str, float]:
    model.eval()

    total_losses = []
    hole_losses = []
    valid_losses = []
    perceptual_losses = []
    metric_dicts = []

    for idx, batch in enumerate(tqdm(loader, desc="Val", leave=False)):
        model_input = batch["input"].to(device)
        masked_rgb = batch["masked"].to(device)
        mask = batch["mask"].to(device)
        gt = batch["gt"].to(device)

        pred_rgb = model(model_input)
        reconstructed = blend_prediction_with_known_region(pred_rgb, masked_rgb, mask)

        loss, stats = criterion(pred_rgb, gt, mask)

        total_losses.append(stats["loss_total"])
        hole_losses.append(stats["loss_hole"])
        valid_losses.append(stats["loss_valid"])
        perceptual_losses.append(stats["loss_perceptual"])

        if max_metric_samples is None or idx < max_metric_samples:
            batch_size = reconstructed.shape[0]
            for b in range(batch_size):
                metrics = compute_all_metrics(
                    reconstructed[b].cpu(), gt[b].cpu(), mask[b].cpu()
                )
                metric_dicts.append(metrics)

    val_psnr = float(np.mean([m["psnr"] for m in metric_dicts]))
    val_ssim = float(np.mean([m["ssim"] for m in metric_dicts]))
    val_lpips = float(np.mean([m["lpips"] for m in metric_dicts]))

    return {
        "loss_total": float(np.mean(total_losses)),
        "loss_hole": float(np.mean(hole_losses)),
        "loss_valid": float(np.mean(valid_losses)),
        "loss_perceptual": float(np.mean(perceptual_losses)),
        "psnr": val_psnr,
        "ssim": val_ssim,
        "lpips": val_lpips,
    }
