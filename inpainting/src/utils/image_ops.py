from pathlib import Path
from typing import List

import torchvision.transforms.functional as TF
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(root_dir: str | Path) -> List[Path]:
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Image root directory does not exist: {root_dir}")

    image_paths = [p for p in root_dir.rglob("*") if is_image_file(p)]
    image_paths = sorted(image_paths)

    if not image_paths:
        raise RuntimeError(f"No image files found in: {root_dir}")

    return image_paths


def load_rgb_image(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        return img.convert("RGB")


def ensure_min_size(image: Image.Image, min_size: int) -> Image.Image:
    width, height = image.size

    if width >= min_size and height >= min_size:
        return image

    scale = max(min_size / width, min_size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def crop_image(image: Image.Image, top: int, left: int, crop_size: int) -> Image.Image:
    return TF.crop(image, top=top, left=left, height=crop_size, width=crop_size)


def to_tensor(image: Image.Image):
    return TF.to_tensor(image)
