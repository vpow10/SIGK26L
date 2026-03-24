import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.blocks = nn.ModuleList(
            [
                vgg[:4],  # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:16],  # relu3_3
            ]
        )
        self.blocks.eval()

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        features = []

        for block in self.blocks:
            x = block(x)
            features.append(x)

        return features


def sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    dtype = x.dtype
    channels = x.shape[1]

    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)

    kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)

    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)

    grad_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
    return grad_x, grad_y


class InpaintingLoss(nn.Module):
    """
    total = hole_l1
        + valid_weight * valid_l1
        + perceptual_weight * perceptual_loss
        + gradient_weight * gradient_loss
    """

    def __init__(
        self,
        hole_weight: float = 1.0,
        valid_weight: float = 0.1,
        perceptual_weight: float = 0.05,
        gradient_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.perceptual_weight = perceptual_weight
        self.gradient_weight = gradient_weight

        self.l1 = nn.L1Loss()
        self.perceptual_net = VGGPerceptualFeatures()

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

        hole_l1 = self.l1(pred_hole, target_hole)
        valid_l1 = self.l1(pred_valid, target_valid)

        pred_feats = self.perceptual_net(pred)
        target_feats = self.perceptual_net(target)

        perceptual_loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            perceptual_loss = perceptual_loss + self.l1(pf, tf)

        pred_gx, pred_gy = sobel_gradients(pred)
        target_gx, target_gy = sobel_gradients(target)

        gradient_loss = self.l1(pred_gx * hole_region, target_gx * hole_region)
        gradient_loss = gradient_loss + self.l1(
            pred_gy * hole_region, target_gy * hole_region
        )

        total_loss = (
            self.hole_weight * hole_l1
            + self.valid_weight * valid_l1
            + self.perceptual_weight * perceptual_loss
            + self.gradient_weight * gradient_loss
        )

        stats = {
            "loss_total": float(total_loss.detach().item()),
            "loss_hole": float(hole_l1.detach().item()),
            "loss_valid": float(valid_l1.detach().item()),
            "loss_perceptual": float(perceptual_loss.detach().item()),
            "loss_gradient": float(gradient_loss.detach().item()),
        }
        return total_loss, stats
