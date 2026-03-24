import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch

from src.models.build import build_model
from src.utils.config import load_config
from src.utils.data import build_denoising_dataloader
from src.utils.device import get_device
from src.utils.seed import set_seed


def main() -> None:
    config = load_config("configs/base.yaml")
    set_seed(config["seed"])

    device = get_device(config["train"]["device"])
    sigma = config["noise"]["sigma"]

    loader = build_denoising_dataloader(
        config=config,
        split="train",
        sigma=sigma,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    noisy = batch["noisy"].to(device)  # [B, 3, H, W]
    gt = batch["gt"].to(device)        # [B, 3, H, W]

    model = build_model(config).to(device)
    model.eval()

    with torch.no_grad():
        pred = model(noisy)

    print("=== Model Smoke Test ===")
    print(f"Device:        {device}")
    print(f"Sigma:         {sigma}")
    print(f"Noisy shape:   {tuple(noisy.shape)}")
    print(f"GT shape:      {tuple(gt.shape)}")
    print(f"Pred shape:    {tuple(pred.shape)}")
    print(f"Pred min/max:  {pred.min().item():.4f} / {pred.max().item():.4f}")

    assert pred.shape == gt.shape, "Prediction shape must match GT shape"

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
