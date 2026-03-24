from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from skimage.util import random_noise
from src.utils.image_ops import (
    crop_image,
    ensure_min_size,
    list_image_files,
    load_rgb_image,
    to_tensor,
)
from src.utils.splits import load_split_file
from torch.utils.data import Dataset


class DenoisingDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        sigma: float = 0.01,
        split: str = "train",
        split_file: Optional[str | Path] = None,
        seed: int = 47,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.sigma = sigma
        self.split = split
        self.seed = seed

        if split_file is not None:
            self.image_paths: List[Path] = load_split_file(split_file, self.root_dir)
        else:
            self.image_paths = list_image_files(self.root_dir)

        if len(self.image_paths) == 0:
            raise RuntimeError("Dataset is empty.")

        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported split: {self.split}.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_rng(self, idx: int) -> np.random.Generator:
        if self.split == "train":
            return np.random.default_rng()

        return np.random.default_rng(self.seed + idx)

    def _get_crop_params(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        if width < self.image_size or height < self.image_size:
            raise ValueError(
                f"Image size ({width}, {height}) is smaller than crop size ({self.image_size})"
            )

        if self.split == "train":
            top = int(rng.integers(0, height - self.image_size + 1))
            left = int(rng.integers(0, width - self.image_size + 1))
        else:
            top = (height - self.image_size) // 2
            left = (width - self.image_size) // 2

        return top, left

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | float]:
        image_path = self.image_paths[idx]
        image = load_rgb_image(image_path)
        image = ensure_min_size(image, self.image_size)

        rng = self._get_rng(idx)

        width, height = image.size
        top, left = self._get_crop_params(width, height, rng)
        image = crop_image(image, top=top, left=left, crop_size=self.image_size)

        gt = to_tensor(image)

        image_np = gt.permute(1, 2, 0).numpy()
        noise_seed = None if self.split == "train" else int(self.seed + idx)
        noisy_np = random_noise(
            image_np,
            mode="gaussian",
            var=self.sigma ** 2,
            rng=noise_seed,
            clip=True,
        )
        noisy = torch.from_numpy(noisy_np.astype(np.float32)).permute(2, 0, 1)

        return {
            "noisy": noisy,
            "gt": gt,
            "sigma": self.sigma,
            "path": str(image_path),
        }
