from pathlib import Path

from src.datasets.inpainting_dataset import InpaintingDataset
from torch.utils.data import DataLoader


def resolve_root_dir(config: dict, split: str) -> Path:
    if split in {"train", "val"}:
        return Path(config["paths"]["data_train_root"])
    if split == "test":
        return Path(config["paths"]["data_test_root"])
    raise ValueError(f"Unsupported split: {split}")


def build_inpainting_dataloader(
    config: dict,
    split: str,
    mask_size: int,
    batch_size: int = 1,
    shuffle: bool = False,
) -> DataLoader:
    root_dir = resolve_root_dir(config, split)
    split_file = Path(config["paths"]["splits"]) / f"{split}.txt"

    dataset = InpaintingDataset(
        root_dir=root_dir,
        image_size=config["data"]["image_size"],
        mask_size=mask_size,
        split=split,
        split_file=split_file,
        seed=config["seed"],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config["train"]["num_workers"],
        pin_memory=False,
    )
    return loader
