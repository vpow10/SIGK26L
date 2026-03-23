import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.build import build_model
from src.utils.config import load_config
from src.utils.data import build_inpainting_dataloader
from src.utils.device import get_device
from src.utils.reconstruction import blend_prediction_with_known_region
from src.utils.seed import set_seed


def main() -> None:
    config = load_config("configs/base.yaml")
    set_seed(config["seed"])

    device = get_device(config["train"]["device"])

    loader = build_inpainting_dataloader(
        config=config,
        split="train",
        mask_size=config["mask"]["size"],
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    model_input = batch["input"].to(device)  # [B, 4, H, W]
    masked_rgb = batch["masked"].to(device)  # [B, 3, H, W]
    mask = batch["mask"].to(device)  # [B, 1, H, W]
    gt = batch["gt"].to(device)  # [B, 3, H, W]

    model = build_model(config).to(device)
    model.eval()

    with torch.no_grad():
        pred_rgb = model(model_input)
        reconstructed = blend_prediction_with_known_region(
            pred_rgb=pred_rgb,
            masked_rgb=masked_rgb,
            mask=mask,
        )

    print("=== Model Smoke Test ===")
    print(f"Device:           {device}")
    print(f"Input shape:      {tuple(model_input.shape)}")
    print(f"Masked shape:     {tuple(masked_rgb.shape)}")
    print(f"Mask shape:       {tuple(mask.shape)}")
    print(f"GT shape:         {tuple(gt.shape)}")
    print(f"Pred RGB shape:   {tuple(pred_rgb.shape)}")
    print(f"Recon shape:      {tuple(reconstructed.shape)}")
    print(
        f"Pred min/max:     {pred_rgb.min().item():.4f} / {pred_rgb.max().item():.4f}"
    )
    print(
        f"Recon min/max:    {reconstructed.min().item():.4f} / {reconstructed.max().item():.4f}"
    )

    assert pred_rgb.shape == gt.shape, "Prediction shape must match GT shape"
    assert reconstructed.shape == gt.shape, "Reconstruction shape must match GT shape"

    print("Model smoke test passed.")


if __name__ == "__main__":
    main()
