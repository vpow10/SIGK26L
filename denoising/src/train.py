import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch

from src.models.build import build_model
from src.utils.checkpoints import save_checkpoint
from src.utils.config import load_config
from src.utils.data import build_denoising_dataloader
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
    sigma = config["noise"]["sigma"]
    experiment_name = f"unet_sigma{str(sigma).replace('.', '')}"

    checkpoints_dir = ensure_dir(Path(config["paths"]["checkpoints"]) / experiment_name)
    logs_dir = ensure_dir(Path(config["paths"]["logs"]) / experiment_name)
    figures_dir = ensure_dir(Path(config["paths"]["figures"]) / experiment_name)

    train_loader = build_denoising_dataloader(
        config=config,
        split="train",
        sigma=sigma,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
    )

    val_loader = build_denoising_dataloader(
        config=config,
        split="val",
        sigma=sigma,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    model = build_model(config).to(device)

    loss_name = config.get("loss", {}).get("name", "l1")
    if loss_name == "l1":
        criterion = torch.nn.L1Loss()
    elif loss_name == "mse":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_lpips": [],
    }

    best_val_psnr = float("-inf")

    print("=== Training start ===")
    print(f"Device: {device}")
    print(f"Experiment: {experiment_name}")
    print(f"Epochs: {config['train']['epochs']}")
    print(f"Sigma: {sigma}")

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

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_psnr"].append(val_stats["psnr"])
        history["val_ssim"].append(val_stats["ssim"])
        history["val_lpips"].append(val_stats["lpips"])

        print(
            f"Train loss: {train_stats['loss']:.6f} | "
            f"Val loss: {val_stats['loss']:.6f} | "
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
