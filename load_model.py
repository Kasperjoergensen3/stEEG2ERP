import torch
from pathlib import Path
import yaml
from model import SplitLatentModelWithVariance

MODELS_DIR = Path("models/")


def load_model(model_name, device="cpu", chkpt=None):
    model_dir = MODELS_DIR / model_name

    # load config.ymal
    with open(model_dir / "config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # load model
    model_module = SplitLatentModelWithVariance
    model = model_module(
        config["in_channels"]["value"],
        config["channels"]["value"],
        config["latent_dim"]["value"],
        config["num_layers"]["value"],
        config["kernel_size"]["value"],
        config["recon_type"]["value"],
        config["content_cosine"]["value"],
    )

    # load state_dict
    if chkpt:
        try:
            state_dict = torch.load(
                model_dir / f"{chkpt}.pt", map_location=torch.device(device)
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint {chkpt} not found in {model_dir}")
    else:
        model_files = list(model_dir.glob("*.pt"))
        if len(model_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {model_dir}")
        elif len(model_files) > 1:
            raise ValueError(
                f"Multiple .pt files found in {model_dir}. Please specify a checkpoint."
            )
        else:
            state_dict = torch.load(model_files[0], map_location=torch.device(device))
    model.load_state_dict(state_dict)

    # return model
    return model, config
