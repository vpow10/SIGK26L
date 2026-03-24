from pathlib import Path

from src.datasets.denoising_dataset import DenoisingDataset
from torch.utils.data import DataLoader


def resolve_root_dir(config: dict, split: str) -> Path:
    if split in {"train", "val"}:
        return Path(config["paths"]["data_train_root"])
    if split == "test":
        return Path(config["paths"]["data_test_root"])
    raise ValueError(f"Unsupported split: {split}")


def build_denoising_dataloader(
    config: dict,
    split: str,
    sigma: float,
    batch_size: int = 1,
    shuffle: bool = False,
) -> DataLoader:
    root_dir = resolve_root_dir(config, split)
    split_file = Path(config["paths"]["splits"]) / f"{split}.txt"

    dataset = DenoisingDataset(
        root_dir=root_dir,
        image_size=config["data"]["image_size"],
        sigma=sigma,
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
