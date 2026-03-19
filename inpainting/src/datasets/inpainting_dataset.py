from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from src.utils.image_ops import (
    crop_image,
    ensure_min_size,
    list_image_files,
    load_rgb_image,
    to_tensor,
)
from src.utils.masks import apply_mask, generate_square_mask
from src.utils.splits import load_split_file
from torch.utils.data import Dataset


class InpaintingDataset(Dataset):
    """
    Dataset for image inpainting.

    Returns:
        {
            "input":  [4, H, W]  -> masked RGB + mask
            "gt":     [3, H, W]  -> clean target
            "masked": [3, H, W]  -> corrupted image
            "mask":   [1, H, W]  -> binary mask
            "path":   str        -> source file path
            "mask_top": int
            "mask_left": int
            "mask_size": int
        }
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        mask_size: int = 32,
        split: str = "train",
        split_file: Optional[str | Path] = None,
        seed: int = 47,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.mask_size = mask_size
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
                f"Image size ({width}, {height}) is smaller than crop size ({self.image_size}"
            )

        if self.split == "train":
            top = int(rng.integers(0, height - self.image_size + 1))
            left = int(rng.integers(0, width - self.image_size + 1))
        else:
            top = (height - self.image_size) // 2
            left = (width - self.image_size) // 2

        return top, left

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        image_path = self.image_paths[idx]
        image = load_rgb_image(image_path)
        image = ensure_min_size(image, self.image_size)

        rng = self._get_rng(idx)

        width, height = image.size
        top, left = self._get_crop_params(width, height, rng)
        image = crop_image(image, top=top, left=left, crop_size=self.image_size)

        gt = to_tensor(image)  # [3, H, W], float in [0, 1]

        mask, mask_meta = generate_square_mask(
            height=self.image_size,
            width=self.image_size,
            patch_size=self.mask_size,
            rng=rng,
        )

        masked = apply_mask(gt, mask)
        model_input = torch.cat([masked, mask], dim=0)  # [4, H, W]

        return {
            "input": model_input,
            "gt": gt,
            "masked": masked,
            "mask": mask,
            "path": str(image_path),
            "mask_top": mask_meta["top"],
            "mask_left": mask_meta["left"],
            "mask_size": mask_meta["patch_size"],
        }
