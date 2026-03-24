from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_history(
    history: Dict[str, List[float]], output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L1 loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_psnr"], label="val_psnr")
    axes[1].plot(epochs, history["val_ssim"], label="val_ssim")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
