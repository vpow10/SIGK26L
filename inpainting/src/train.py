import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.losses.losses import InpaintingTotalLoss
from src.models.build import build_model
from src.utils.checkpoints import save_checkpoint
from src.utils.config import load_config
from src.utils.data import build_inpainting_dataloader
from src.utils.device import get_device
from src.utils.io import ensure_dir
from src.utils.seed import set_seed
from src.utils.train_eval_loops import train_one_epoch, validate_one_epoch
from src.utils.training_viz import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_val_metric_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = get_device(config["train"]["device"])
    mask_size = config["mask"]["size"]
    experiment_name = f"{config['model']['name']}_mask{mask_size}"

    checkpoints_dir = ensure_dir(Path(config["paths"]["checkpoints"]) / experiment_name)
    logs_dir = ensure_dir(Path(config["paths"]["logs"]) / experiment_name)
    figures_dir = ensure_dir(Path(config["paths"]["figures"]) / experiment_name)

    train_loader = build_inpainting_dataloader(
        config=config,
        split="train",
        mask_size=mask_size,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )

    val_loader = build_inpainting_dataloader(
        config=config,
        split="val",
        mask_size=mask_size,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    model = build_model(config).to(device)

    criterion = InpaintingTotalLoss(
        hole_weight=config["loss"]["hole_weight"],
        valid_weight=config["loss"]["valid_weight"],
        perceptual_weight=config["loss"]["perceptual_weight"],
        style_weight=config["loss"]["style_weight"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
    )

    history = {
        "train_loss": [],
        "train_hole_loss": [],
        "train_valid_loss": [],
        "train_perceptual_loss": [],
        "train_style_loss": [],
        "val_loss": [],
        "val_hole_loss": [],
        "val_valid_loss": [],
        "val_perceptual_loss": [],
        "val_style_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_lpips": [],
    }

    best_val_psnr = float("-inf")

    print("=== Training start ===")
    print(f"Device: {device}")
    print(f"Experiment: {experiment_name}")
    print(f"Epochs: {config['train']['epochs']}")
    print(f"Mask size: {mask_size}")

    for epoch in range(1, config["train"]["epochs"] + 1):
        print(f"\nEpoch [{epoch}/{config['train']['epochs']}]")

        train_stats = train_one_epoch(
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
            max_metric_samples=args.max_val_metric_samples,
        )

        history["train_loss"].append(train_stats["loss_total"])
        history["train_hole_loss"].append(train_stats["loss_hole"])
        history["train_valid_loss"].append(train_stats["loss_valid"])
        history["train_perceptual_loss"].append(train_stats["loss_perceptual"])
        history["train_style_loss"].append(train_stats["loss_style"])

        history["val_loss"].append(val_stats["loss_total"])
        history["val_hole_loss"].append(val_stats["loss_hole"])
        history["val_valid_loss"].append(val_stats["loss_valid"])
        history["val_perceptual_loss"].append(val_stats["loss_perceptual"])
        history["val_style_loss"].append(val_stats["loss_style"])
        history["val_psnr"].append(val_stats["psnr"])
        history["val_ssim"].append(val_stats["ssim"])
        history["val_lpips"].append(val_stats["lpips"])

        print(
            f"Train loss: {train_stats['loss_total']:.6f} | "
            f"Val loss: {val_stats['loss_total']:.6f} | "
            f"Val PSNR: {val_stats['psnr']:.4f} | "
            f"Val SSIM: {val_stats['ssim']:.4f} | "
            f"Val LPIPS: {val_stats['lpips']:.4f}"
        )

        last_ckpt_path = checkpoints_dir / "last.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_psnr=best_val_psnr,
            config=config,
            output_path=last_ckpt_path,
        )

        if val_stats["psnr"] > best_val_psnr:
            best_val_psnr = val_stats["psnr"]
            best_ckpt_path = checkpoints_dir / "best.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_psnr=best_val_psnr,
                config=config,
                output_path=best_ckpt_path,
            )
            print(f"Saved new best checkpoint: {best_ckpt_path}")

        with open(logs_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        plot_training_history(
            history=history,
            output_path=figures_dir / "training_curves.png",
        )

    print("\n=== Training complete ===")
    print(f"Best val PSNR: {best_val_psnr:.4f}")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Logs: {logs_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
