import torch
import torch.nn as nn
import torchvision.models as models


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


class VGGPerceptualAndStyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)])  # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)])  # relu3_3

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.l1 = nn.L1Loss()

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        pred_f1 = self.slice1(pred)
        tgt_f1 = self.slice1(target)

        pred_f2 = self.slice2(pred_f1)
        tgt_f2 = self.slice2(tgt_f1)

        pred_f3 = self.slice3(pred_f2)
        tgt_f3 = self.slice3(tgt_f2)

        perc_loss = (
            self.l1(pred_f1, tgt_f1)
            + self.l1(pred_f2, tgt_f2)
            + self.l1(pred_f3, tgt_f3)
        )

        style_loss = (
            self.l1(self.gram_matrix(pred_f1), self.gram_matrix(tgt_f1))
            + self.l1(self.gram_matrix(pred_f2), self.gram_matrix(tgt_f2))
            + self.l1(self.gram_matrix(pred_f3), self.gram_matrix(tgt_f3))
        )

        return perc_loss, style_loss


class InpaintingTotalLoss(nn.Module):
    """Combines L1, Perceptual, and Style Loss"""

    def __init__(
        self,
        hole_weight: float = 1.0,
        valid_weight: float = 0.1,
        perceptual_weight: float = 0.1,
        style_weight: float = 120.0,
    ):
        super().__init__()
        self.l1_loss = WeightedL1InpaintingLoss(hole_weight, valid_weight)
        self.vgg_loss = VGGPerceptualAndStyleLoss()

        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        l1_total, stats = self.l1_loss(pred, target, mask)

        comp = pred * mask + target * (1.0 - mask)
        p_loss, s_loss = self.vgg_loss(comp, target)

        total_loss = (
            l1_total + (self.perceptual_weight * p_loss) + (self.style_weight * s_loss)
        )

        stats["loss_perceptual"] = float(p_loss.detach().item())
        stats["loss_style"] = float(s_loss.detach().item())
        stats["loss_total"] = float(total_loss.detach().item())

        return total_loss, stats
