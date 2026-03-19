# Image Inpainting Mini Project

A small research-oriented project on image inpainting using PyTorch.

## Goal
Reconstruct missing image regions in DIV2K images and compare a lightweight learned method against OpenCV Telea inpainting.

## Planned experiments
- Mask sizes: 3x3 and 32x32
- Baseline: OpenCV `INPAINT_TELEA`
- Model: lightweight U-Net
- Metrics: PSNR, SSIM, LPIPS

## Structure
- `src/` - source code
- `configs/` - experiment configs
- `data/` - raw and processed data
- `results/` - checkpoints, logs, figures
- `scripts/` - helper scripts