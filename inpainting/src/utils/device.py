import torch


def get_device(requested_device: str = "cuda") -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
