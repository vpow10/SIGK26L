from pathlib import Path
from typing import List


def save_split_file(
    paths: List[Path], root_dir: str | Path, output_path: str | Path
) -> None:
    root_dir = Path(root_dir).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    relative_paths = [p.resolve().relative_to(root_dir).as_posix() for p in paths]

    with open(output_path, "w", encoding="utf-8") as f:
        for rel_path in relative_paths:
            f.write(rel_path + "\n")


def load_split_file(split_file: str | Path, root_dir: str | Path) -> List[Path]:
    split_file = Path(split_file)
    root_dir = Path(root_dir)

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    paths = [root_dir / line for line in lines]

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Some files listed in split file do not exist. "
            f"First missing example: {missing[0]}"
        )

    return paths
