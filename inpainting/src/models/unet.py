import torch
import torch.nn as nn
from src.models.blocks import ConvBlock, DownBlock, UpBlock


class UNet(nn.Module):
    """
    Input:
        [B, 4, H, W]  -> masked RGB + binary mask
    Output:
        [B, 3, H, W]  -> predicted RGB content
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.bottleneck = DownBlock(c4, c5)

        self.dec1 = UpBlock(c5, c4, c4)
        self.dec2 = UpBlock(c4, c3, c3)
        self.dec3 = UpBlock(c3, c2, c2)
        self.dec4 = UpBlock(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        xb = self.bottleneck(x4)

        y1 = self.dec1(xb, x4)
        y2 = self.dec2(y1, x3)
        y3 = self.dec3(y2, x2)
        y4 = self.dec4(y3, x1)

        out = self.out_conv(y4)
        return out
