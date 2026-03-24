import numpy as np
import torch
from skimage.restoration import denoise_bilateral


def bilateral_denoise(noisy: torch.Tensor) -> torch.Tensor:
    image_np = noisy.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()

    denoised_np = denoise_bilateral(
        image_np,
        channel_axis=-1,
    )

    denoised = torch.from_numpy(denoised_np.astype(np.float32)).permute(2, 0, 1)
    return denoised
