from pathlib import Path

import matplotlib.pyplot as plt
import torch
from src.utils.visualization import mask_to_image_np, tensor_to_image_np


def save_comparison_figure(
    gt: torch.Tensor,
    masked: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    output_path: str | Path,
    title: str = "",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gt_np = tensor_to_image_np(gt)
    masked_np = tensor_to_image_np(masked)
    pred_np = tensor_to_image_np(pred)
    mask_np = mask_to_image_np(mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

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
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
