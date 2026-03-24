import argparse
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch

from src.baselines.bilateral import bilateral_denoise
from src.utils.config import load_config
from src.utils.data import build_denoising_dataloader
from src.utils.seed import set_seed


SIGMA_COLOR_VALUES = [0.02, 0.05, 0.10, 0.15, 0.20]
SIGMA_SPATIAL_VALUES = [1, 2, 3, 5]
NOISE_SIGMAS = [0.01, 0.03]


def compute_psnr_np(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    mse = np.mean((pred_np - gt_np) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def evaluate_params(loader, sigma_color: float, sigma_spatial: float, n_images: int) -> float:
    psnrs = []
    for idx, batch in enumerate(loader):
        if idx >= n_images:
            break
        noisy = batch["noisy"][0]
        gt = batch["gt"][0]

        pred = bilateral_denoise(noisy, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

        psnrs.append(compute_psnr_np(tensor_to_np(pred), tensor_to_np(gt)))

    return float(np.mean(psnrs))


def print_table(results: list, noise_sigma: float) -> tuple:
    results_sorted = sorted(results, key=lambda x: x["psnr"], reverse=True)
    best = results_sorted[0]

    col_w = 14
    header = (
        f"{'sigma_color':>{col_w}}  "
        f"{'sigma_spatial':>{col_w}}  "
        f"{'PSNR (dB)':>{col_w}}"
    )
    sep = "-" * len(header)

    print(f"\n=== Bilateral tuning — noise sigma={noise_sigma} ===")
    print(header)
    print(sep)

    for r in results_sorted:
        marker = " <-- best" if r is best else ""
        print(
            f"{r['sigma_color']:>{col_w}.2f}  "
            f"{r['sigma_spatial']:>{col_w}}  "
            f"{r['psnr']:>{col_w}.4f}"
            f"{marker}"
        )

    print(sep)
    print(
        f"Best: sigma_color={best['sigma_color']}, "
        f"sigma_spatial={best['sigma_spatial']}, "
        f"PSNR={best['psnr']:.4f} dB"
    )
    return best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_images",
        type=int,
        default=30,
        help="Number of val images to evaluate per combination (default: 30).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    n_combinations = len(SIGMA_COLOR_VALUES) * len(SIGMA_SPATIAL_VALUES)
    total_runs = n_combinations * len(NOISE_SIGMAS)
    print(f"Grid: {len(SIGMA_COLOR_VALUES)} sigma_color x {len(SIGMA_SPATIAL_VALUES)} sigma_spatial")
    print(f"Noise sigmas: {NOISE_SIGMAS}")
    print(f"Total runs: {total_runs}, each on {args.n_images} val images")
    print("(PSNR only — fast. Final evaluation with all metrics via evaluate.py)")

    best_per_sigma = {}

    for noise_sigma in NOISE_SIGMAS:
        loader = build_denoising_dataloader(
            config=config,
            split="val",
            sigma=noise_sigma,
            batch_size=1,
            shuffle=False,
        )

        results = []
        for sigma_color, sigma_spatial in product(SIGMA_COLOR_VALUES, SIGMA_SPATIAL_VALUES):
            psnr = evaluate_params(loader, sigma_color, sigma_spatial, args.n_images)
            results.append(
                {
                    "sigma_color": sigma_color,
                    "sigma_spatial": sigma_spatial,
                    "psnr": psnr,
                }
            )

        best = print_table(results, noise_sigma)
        best_per_sigma[noise_sigma] = best


if __name__ == "__main__":
    main()
