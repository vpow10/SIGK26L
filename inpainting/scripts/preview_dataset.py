import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.datasets.inpainting_dataset import InpaintingDataset
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.visualization import mask_to_image_np, tensor_to_image_np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument("--mask_size", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    return parser.parse_args()


def resolve_root_dir(config: dict, split: str) -> Path:
    if split in {"train", "val"}:
        return Path(config["paths"]["data_train_root"])
    if split == "test":
        return Path(config["paths"]["data_test_root"])
    raise ValueError(f"Unsupported split: {split}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    mask_size = args.mask_size if args.mask_size is not None else config["mask"]["size"]

    root_dir = resolve_root_dir(config, args.split)
    split_file = Path(config["paths"]["splits"]) / f"{args.split}.txt"

    dataset = InpaintingDataset(
        root_dir=root_dir,
        image_size=config["data"]["image_size"],
        mask_size=mask_size,
        split=args.split,
        split_file=split_file,
        seed=config["seed"],
    )

    num_samples = min(args.num_samples, len(dataset))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for row_idx in range(num_samples):
        sample = dataset[row_idx]

        gt_np = tensor_to_image_np(sample["gt"])
        masked_np = tensor_to_image_np(sample["masked"])
        mask_np = mask_to_image_np(sample["mask"])

        ax0, ax1, ax2 = axes[row_idx]
        ax0.imshow(gt_np)
        ax0.set_title("Ground Truth")
        ax0.axis("off")

        ax1.imshow(masked_np)
        ax1.set_title(
            f"Masked | size={sample['mask_size']} | "
            f"top={sample['mask_top']}, left={sample['mask_left']}"
        )
        ax1.axis("off")

        ax2.imshow(mask_np, cmap="gray")
        ax2.set_title("Mask")
        ax2.axis("off")

    plt.tight_layout()

    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_path = figures_dir / f"dataset_preview_{args.split}_mask{mask_size}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved preview to: {save_path}")


if __name__ == "__main__":
    main()
