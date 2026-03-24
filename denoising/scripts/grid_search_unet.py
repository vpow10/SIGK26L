import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.models.build import build_model
from src.utils.config import load_config
from src.utils.data import build_denoising_dataloader
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.train_eval_loops import train_one_epoch, validate_one_epoch


LEARNING_RATES = [0.0001, 0.001]
BASE_CHANNELS = [32, 64]
LOSSES = ["l1", "mse"]


def get_criterion(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {loss_name}")


def make_config(base_config: dict, lr: float, base_channels: int, sigma: float) -> dict:
    import copy
    cfg = copy.deepcopy(base_config)
    cfg["train"]["learning_rate"] = lr
    cfg["model"]["base_channels"] = base_channels
    cfg["noise"]["sigma"] = sigma
    return cfg


def run_one(
    base_config: dict,
    lr: float,
    base_channels: int,
    loss_name: str,
    sigma: float,
    epochs: int,
    device: torch.device,
    train_loader,
    val_loader,
) -> dict:
    cfg = make_config(base_config, lr, base_channels, sigma)
    set_seed(cfg["seed"])

    model = build_model(cfg).to(device)
    criterion = get_criterion(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_psnr = float("-inf")
    best_val_ssim = 0.0
    best_val_lpips = 1.0

    for epoch in range(1, epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if val_stats["psnr"] > best_val_psnr:
            best_val_psnr = val_stats["psnr"]
            best_val_ssim = val_stats["ssim"]
            best_val_lpips = val_stats["lpips"]
            best_epoch = epoch

    return {
        "lr": lr,
        "base_channels": base_channels,
        "loss": loss_name,
        "best_epoch": best_epoch,
        "val_psnr": best_val_psnr,
        "val_ssim": best_val_ssim,
        "val_lpips": best_val_lpips,
    }


def print_results_table(results: list, sigma: float, epochs: int) -> dict:
    results_sorted = sorted(results, key=lambda r: r["val_psnr"], reverse=True)
    best = results_sorted[0]

    print(f"\n{'=' * 78}")
    print(f"U-Net Grid Search Results  |  sigma={sigma}  |  {epochs} epochs each")
    print(f"{'=' * 78}")
    print(
        f"{'lr':>8}  {'base_ch':>8}  {'loss':>5}  "
        f"{'best_ep':>8}  {'Val PSNR':>10}  {'Val SSIM':>10}  {'Val LPIPS':>10}"
    )
    print("-" * 78)

    for r in results_sorted:
        marker = " <-- best" if r is best else ""
        print(
            f"{r['lr']:>8}  "
            f"{r['base_channels']:>8}  "
            f"{r['loss']:>5}  "
            f"{r['best_epoch']:>8}  "
            f"{r['val_psnr']:>10.4f}  "
            f"{r['val_ssim']:>10.4f}  "
            f"{r['val_lpips']:>10.4f}"
            f"{marker}"
        )

    print("-" * 78)
    print(
        f"Best: lr={best['lr']}, base_channels={best['base_channels']}, "
        f"loss={best['loss']}  →  PSNR={best['val_psnr']:.4f} dB"
    )
    return best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Epochs per combination (default: 15). Use fewer for a quick scan.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.01,
        help="Noise sigma to use for grid search (default: 0.01).",
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    device = get_device(base_config["train"]["device"])

    n_combinations = len(LEARNING_RATES) * len(BASE_CHANNELS) * len(LOSSES)
    print(f"Grid: {LEARNING_RATES} x {BASE_CHANNELS} x {LOSSES}")
    print(f"Total combinations: {n_combinations}")
    print(f"Epochs per run: {args.epochs}  |  Noise sigma: {args.sigma}")
    print(f"Device: {device}")
    print("Starting grid search...\n")

    train_loader = build_denoising_dataloader(
        config=base_config,
        split="train",
        sigma=args.sigma,
        batch_size=base_config["train"]["batch_size"],
        shuffle=True,
    )
    val_loader = build_denoising_dataloader(
        config=base_config,
        split="val",
        sigma=args.sigma,
        batch_size=base_config["train"]["batch_size"],
        shuffle=False,
    )

    results = []
    run_idx = 0

    for lr in LEARNING_RATES:
        for base_channels in BASE_CHANNELS:
            for loss_name in LOSSES:
                run_idx += 1
                print(
                    f"[{run_idx}/{n_combinations}] "
                    f"lr={lr}  base_channels={base_channels}  loss={loss_name}"
                )

                result = run_one(
                    base_config=base_config,
                    lr=lr,
                    base_channels=base_channels,
                    loss_name=loss_name,
                    sigma=args.sigma,
                    epochs=args.epochs,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                )
                results.append(result)
                print(f"    → Val PSNR: {result['val_psnr']:.4f} dB (best at epoch {result['best_epoch']})")

if __name__ == "__main__":
    main()
