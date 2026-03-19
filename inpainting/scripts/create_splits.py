import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.image_ops import list_image_files
from src.utils.splits import save_split_file


def main() -> None:
    config = load_config("configs/base.yaml")

    train_root = Path(config["paths"]["data_train_root"])
    test_root = Path(config["paths"]["data_test_root"])
    splits_dir = Path(config["paths"]["splits"])

    train_ratio = config["data"]["train_ratio_within_train_root"]
    val_ratio = config["data"]["val_ratio_within_train_root"]
    seed = config["seed"]

    total_ratio = train_ratio + val_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"Train/val ratios must sum to 1.0, got {total_ratio:.6f}")

    train_all_paths = list_image_files(train_root)
    test_paths = list_image_files(test_root)

    random.seed(seed)
    random.shuffle(train_all_paths)

    n_total_train = len(train_all_paths)
    n_train = int(n_total_train * train_ratio)
    n_val = n_total_train - n_train

    train_paths = train_all_paths[:n_train]
    val_paths = train_all_paths[n_train:]

    save_split_file(train_paths, train_root, splits_dir / "train.txt")
    save_split_file(val_paths, train_root, splits_dir / "val.txt")
    save_split_file(test_paths, test_root, splits_dir / "test.txt")

    print("=== Split generation complete ===")
    print(f"Train root: {train_root}")
    print(f"Test root:  {test_root}")
    print(f"Train-all images: {n_total_train}")
    print(f"Train split:      {len(train_paths)}")
    print(f"Val split:        {len(val_paths)}")
    print(f"Test split:       {len(test_paths)}")
    print(f"Saved to: {splits_dir}")


if __name__ == "__main__":
    main()
