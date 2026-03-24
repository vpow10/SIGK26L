from src.models.unet import UNet


def build_model(config: dict):
    model_name = config.get("model", {}).get("name", "unet")

    if model_name == "unet":
        return UNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            base_channels=config["model"]["base_channels"],
        )

    raise ValueError(f"Unsupported model name: {model_name}")
