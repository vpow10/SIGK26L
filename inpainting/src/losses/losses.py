import torch
import torch.nn as nn


class WeightedL1InpaintingLoss(nn.Module):
    """
    Weighted L1 loss for inpainting.

    loss = hole_weight * L1(pred in missing region, target in missing region)
         + valid_weight * L1(pred in known region, target in known region)
    """

    def __init__(self, hole_weight: float = 1.0, valid_weight: float = 0.1) -> None:
        super().__init__()
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must match, got {pred.shape} vs {target.shape}"
            )
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"mask must be [B, 1, H, W], got {tuple(mask.shape)}")

        hole_region = mask
        valid_region = 1.0 - mask

        pred_hole = pred * hole_region
        target_hole = target * hole_region

        pred_valid = pred * valid_region
        target_valid = target * valid_region

        hole_loss = self.l1(pred_hole, target_hole)
        valid_loss = self.l1(pred_valid, target_valid)

        total_loss = self.hole_weight * hole_loss + self.valid_weight * valid_loss

        stats = {
            "loss_total": float(total_loss.detach().item()),
            "loss_hole": float(hole_loss.detach().item()),
            "loss_valid": float(valid_loss.detach().item()),
        }
        return total_loss, stats
