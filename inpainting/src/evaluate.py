"""
example usage: python src/evaluate.py --method telea --split test --mask_size 32 -> baseline

for unet evaluation, specify the checkpoint path:
python src/evaluate.py \
  --config configs/train_32.yaml \
  --method unet \
  --split test \
  --mask_size 32 \
  --checkpoint results/checkpoints/unet_mask32/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.baselines.telea import telea_inpaint
from src.metrics.metrics import average_metric_dicts, compute_all_metrics
from src.models.build import build_model
from src.utils.checkpoints import load_checkpoint
from src.utils.config import load_config
from src.utils.data import build_inpainting_dataloader
from src.utils.device import get_device
from src.utils.io import ensure_dir, save_tensor_image
from src.utils.reconstruction import blend_prediction_with_known_region
from src.utils.result_viz import save_comparison_figure
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--method", type=str, required=True, choices=["telea", "unet"])
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--mask_size", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_examples", type=int, default=10)
    parser.add_argument("--telea_radius", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = get_device(config["train"]["device"])

    loader = build_inpainting_dataloader(
        config=config,
        split=args.split,
        mask_size=args.mask_size,
        batch_size=1,
        shuffle=False,
    )

    method_name = args.method
    results_root = ensure_dir(
        Path("results") / "eval" / f"{method_name}_mask{args.mask_size}_{args.split}"
    )
    examples_dir = ensure_dir(results_root / "examples")

    model = None
    if method_name == "unet":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when method=unet")

        model = build_model(config).to(device)
        load_checkpoint(model, args.checkpoint, optimizer=None, map_location=device)
        model.eval()

    metric_dicts = []
    per_image_results = []

    for idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {method_name}")):
        if args.max_samples is not None and idx >= args.max_samples:
            break

        gt = batch["gt"][0]
        masked = batch["masked"][0]
        mask = batch["mask"][0]
        model_input = batch["input"].to(device)
        path = batch["path"][0]

        if method_name == "telea":
            pred = telea_inpaint(
                masked_image=masked,
                mask=mask,
                inpaint_radius=args.telea_radius,
            )
        elif method_name == "unet":
            with torch.no_grad():
                pred_rgb = model(model_input)[0]
                pred = blend_prediction_with_known_region(
                    pred_rgb=pred_rgb.unsqueeze(0),
                    masked_rgb=masked.unsqueeze(0).to(device),
                    mask=mask.unsqueeze(0).to(device),
                )[0].cpu()
        else:
            raise ValueError(f"Unsupported method: {method_name}")

        metrics = compute_all_metrics(pred, gt)
        metric_dicts.append(metrics)

        sample_result = {
            "index": idx,
            "path": path,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
        }
        per_image_results.append(sample_result)

        if idx < args.save_examples:
            image_stem = Path(path).stem

            save_tensor_image(gt, examples_dir / f"{idx:03d}_{image_stem}_gt.png")
            save_tensor_image(
                masked, examples_dir / f"{idx:03d}_{image_stem}_masked.png"
            )
            save_tensor_image(pred, examples_dir / f"{idx:03d}_{image_stem}_pred.png")

            save_comparison_figure(
                gt=gt,
                masked=masked,
                mask=mask,
                pred=pred,
                output_path=examples_dir / f"{idx:03d}_{image_stem}_comparison.png",
                title=f"{method_name} | {image_stem} | mask={args.mask_size}",
            )

    aggregate = average_metric_dicts(metric_dicts)

    summary = {
        "method": method_name,
        "split": args.split,
        "mask_size": args.mask_size,
        "num_samples": len(metric_dicts),
        "metrics": aggregate,
        "per_image_results": per_image_results,
    }

    with open(results_root / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final Results ===")
    print(f"Method:     {method_name}")
    print(f"Split:      {args.split}")
    print(f"Mask size:  {args.mask_size}")
    print(f"Samples:    {len(metric_dicts)}")
    print(f"PSNR:       {aggregate['psnr']:.4f}")
    print(f"SSIM:       {aggregate['ssim']:.4f}")
    print(f"LPIPS:      {aggregate['lpips']:.4f}")
    print(f"Saved to:   {results_root}")


if __name__ == "__main__":
    main()
