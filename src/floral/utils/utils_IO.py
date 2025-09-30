import numpy as np
import torch
import os.path as osp
from datetime import datetime
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict


def check_path(path: str, suggestion: str = None) -> None:
    """Check if a path exists, raise an error if it does not."""
    assert path is not None and isinstance(
        path, str
    ), "Path must be a string and not None"
    assert osp.exists(path), f"Path {path} does not exist" + (
        f". Did you mean {suggestion}?" if suggestion else ""
    )


def check_keys(mapping, required_keys):
    """check if the keys are present in the dictionary
    Args:
        mapping (dict): dictionary to check
    """
    assert isinstance(mapping, dict), "target mapping should be a dictionary"
    assert isinstance(required_keys, list) and all(
        isinstance(el, str) for el in required_keys
    ), "required keys should be a list of strings"
    missing_keys = [k for k in required_keys if k not in mapping]
    assert (
        len(missing_keys) == 0
    ), f"Missing keys : {', '.join(missing_keys)} in the dictionary."


def omega_to_dict(cfg: DictConfig, resolve: bool = True) -> Dict[str, Any]:
    """
    Convert an OmegaConf DictConfig (OmegaDict) into a standard Python dict.

    Args:
        cfg (DictConfig): The OmegaConf DictConfig object.
        resolve (bool): Whether to resolve interpolations. Default: True.

    Returns:
        dict: A standard Python dictionary.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    return OmegaConf.to_container(cfg, resolve=resolve)


@rank_zero_only
def printer(message):
    """Print message only on rank 0"""
    print(message)


def t2n(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor to a NumPy array."""
    assert isinstance(tensor, torch.Tensor), "Input must be a PyTorch tensor"
    return tensor.detach().cpu().numpy()


def n2t(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor."""
    assert isinstance(array, np.ndarray), "Input must be a NumPy array"
    return torch.FloatTensor(array)


def check_tensor_blowup(t: torch.Tensor, name: str = "tensor"):
    """
    Check if a tensor has NaN or Inf values and raise an error if so.

    Args:
        t (torch.Tensor): Tensor to check.
        name (str): Optional name to include in the error message.
    """
    if not torch.isfinite(t).all():
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        raise ValueError(
            f"{name} has blown up! "
            f"NaNs: {n_nan}, Infs: {n_inf}, "
            f"min={t.min().item()}, max={t.max().item()}"
        )


def get_logger(name: str):
    """Get a logger with a unique version based on the current time."""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use WandbLogger for better integration with Weights & Biases
    logger = WandbLogger(
        name=f"experiment-{current_time}",  # this appears in wandb dashboard
        project=name,
        log_model=False,  # optional: logs model checkpoints as artifacts
    )
    return logger
