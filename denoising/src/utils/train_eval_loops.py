from typing import Dict

import numpy as np
import torch
from src.metrics.metrics import average_metric_dicts, compute_all_metrics
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    losses = []

    for batch in tqdm(loader, desc="Train", leave=False):
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)

        optimizer.zero_grad()

        pred = model(noisy)
        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    max_metric_samples: int | None = None,
) -> Dict[str, float]:
    model.eval()

    losses = []
    metric_dicts = []

    for idx, batch in enumerate(tqdm(loader, desc="Val", leave=False)):
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)

        pred = model(noisy)
        loss = criterion(pred, gt)
        losses.append(loss.item())

        if max_metric_samples is None or idx < max_metric_samples:
            batch_size = pred.shape[0]
            for b in range(batch_size):
                metrics = compute_all_metrics(pred[b].cpu(), gt[b].cpu())
                metric_dicts.append(metrics)

    avg = average_metric_dicts(metric_dicts)

    return {
        "loss": float(np.mean(losses)),
        "psnr": avg["psnr"],
        "ssim": avg["ssim"],
        "lpips": avg["lpips"],
    }
