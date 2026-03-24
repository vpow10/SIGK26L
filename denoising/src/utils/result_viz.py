from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.metrics.metrics import tensor_to_image_np


def save_comparison_figure(
    gt: torch.Tensor,
    noisy: torch.Tensor,
    pred: torch.Tensor,
    output_path: str | Path,
    title: str = "",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_np = tensor_to_image_np(gt)
    noisy_np = tensor_to_image_np(noisy)
    pred_np = tensor_to_image_np(pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(gt_np)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(noisy_np)
    axes[1].set_title("Noisy Input")
    axes[1].axis("off")

    axes[2].imshow(pred_np)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
