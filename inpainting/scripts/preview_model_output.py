import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.build import build_model
from src.utils.config import load_config
from src.utils.data import build_inpainting_dataloader
from src.utils.device import get_device
from src.utils.reconstruction import blend_prediction_with_known_region
from src.utils.seed import set_seed
from src.utils.visualization import mask_to_image_np, tensor_to_image_np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"]
    )
    parser.add_argument("--mask_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    mask_size = args.mask_size if args.mask_size is not None else config["mask"]["size"]
    device = get_device(config["train"]["device"])

    loader = build_inpainting_dataloader(
        config=config,
        split=args.split,
        mask_size=mask_size,
        batch_size=1,
        shuffle=False,
    )

    batch = next(iter(loader))

    model_input = batch["input"].to(device)
    masked_rgb = batch["masked"].to(device)
    mask = batch["mask"].to(device)
    gt = batch["gt"].to(device)

    model = build_model(config).to(device)
    model.eval()

    with torch.no_grad():
        pred_rgb = model(model_input)
        reconstructed = blend_prediction_with_known_region(pred_rgb, masked_rgb, mask)

    gt_np = tensor_to_image_np(gt[0].cpu())
    masked_np = tensor_to_image_np(masked_rgb[0].cpu())
    pred_np = tensor_to_image_np(pred_rgb[0].cpu())
    recon_np = tensor_to_image_np(reconstructed[0].cpu())
    mask_np = mask_to_image_np(mask[0].cpu())

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(gt_np)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(masked_np)
    axes[1].set_title("Masked")
    axes[1].axis("off")

    axes[2].imshow(mask_np, cmap="gray")
    axes[2].set_title("Mask")
    axes[2].axis("off")

    axes[3].imshow(pred_np)
    axes[3].set_title("Raw Model Output")
    axes[3].axis("off")

    axes[4].imshow(recon_np)
    axes[4].set_title("Blended Reconstruction")
    axes[4].axis("off")

    plt.tight_layout()

    save_path = (
        Path(config["paths"]["figures"])
        / f"untrained_model_preview_{args.split}_mask{mask_size}.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved preview to: {save_path}")


if __name__ == "__main__":
    main()
