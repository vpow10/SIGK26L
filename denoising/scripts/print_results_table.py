import argparse
import json
from pathlib import Path


METRICS = ["sne", "psnr", "ssim", "lpips"]


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*/results.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        sigma = data["sigma"]
        method = data["method"]
        split = data["split"]
        metrics = data.get("metrics", {})

        name = f"{method}_gaussian_sigma{str(sigma).replace('.', '')}_{split}"
        rows.append({"name": name, "metrics": metrics})

    return rows


def print_table(rows: list[dict]) -> None:
    col_name_w = max(len(r["name"]) for r in rows)
    col_w = 10

    header_name = "Metoda"
    header = f"  {header_name:<{col_name_w}}"
    for m in METRICS:
        header += f"  {m.upper():>{col_w}}"

    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for row in rows:
        line = f"  {row['name']:<{col_name_w}}"
        for m in METRICS:
            value = row["metrics"].get(m)
            if value is None:
                line += f"  {'—':>{col_w}}"
            else:
                line += f"  {value:>{col_w}.4f}"
        print(line)

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/eval")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    rows = load_results(results_dir)
    if not rows:
        print("No results.json files found.")
        return

    print_table(rows)


if __name__ == "__main__":
    main()
