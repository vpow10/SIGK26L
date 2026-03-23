import torch


def blend_prediction_with_known_region(
    pred_rgb: torch.Tensor,
    masked_rgb: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if pred_rgb.ndim != 4 or pred_rgb.shape[1] != 3:
        raise ValueError(
            f"Expected pred_rgb shape [B, 3, H, W], got {tuple(pred_rgb.shape)}"
        )
    if masked_rgb.ndim != 4 or masked_rgb.shape[1] != 3:
        raise ValueError(
            f"Expected masked_rgb shape [B, 3, H, W], got {tuple(masked_rgb.shape)}"
        )
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape [B, 1, H, W], got {tuple(mask.shape)}")

    if pred_rgb.shape[0] != masked_rgb.shape[0] or pred_rgb.shape[0] != mask.shape[0]:
        raise ValueError("Batch sizes do not match")

    if (
        pred_rgb.shape[-2:] != masked_rgb.shape[-2:]
        or pred_rgb.shape[-2:] != mask.shape[-2:]
    ):
        raise ValueError("Spatial dimensions do not match")

    reconstructed = masked_rgb * (1.0 - mask) + pred_rgb * mask
    return reconstructed
