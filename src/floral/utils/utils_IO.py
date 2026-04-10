import numpy as np
import torch
import time
import os.path as osp
from datetime import datetime
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict
from functools import wraps


def timeit(func_name=None):
    """Decorator to measure the runtime of a function."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            name = func_name or func.__name__
            print(f"{name} took {elapsed:.6f} seconds")
            return result

        return wrapper

    return decorator


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

    assert isinstance(
        mapping, (dict, DictConfig)
    ), "target mapping should be a dictionary or DictConfig"

    assert isinstance(required_keys, (list, str)) and all(
        isinstance(el, str) for el in required_keys
    ), "required keys should be a list of strings"

    # check missing keys
    if isinstance(required_keys, list):
        missing_keys = [k for k in required_keys if k not in mapping]
        assert (
            len(missing_keys) == 0
        ), f"Missing keys : {', '.join(missing_keys)} in the dictionary."
    elif isinstance(required_keys, str):
        assert (
            required_keys in mapping
        ), f"Missing keys : {required_keys} in the dictionary."
    else:
        raise ValueError(
            f"required keys must be list or string, got {type(required_keys).__name__}"
        )


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


def deep_get(mapping: dict | DictConfig, keys, default=None):
    assert isinstance(
        mapping, (dict, DictConfig)
    ), "target mapping should be a dictionary or DictConfig"
    assert isinstance(keys, list) and len(keys) > 0, "keys must be passed as a list"
    for k in keys:
        if isinstance(mapping, (dict, DictConfig)):
            mapping = mapping.get(k, default)
        else:
            return default
    return mapping


def type_check(data, data_type):
    assert isinstance(
        data, data_type
    ), f"Expected {data_type}, got {type(data).__name__}"


@rank_zero_only
def printer(message):
    """Print message only on rank 0"""
    print(message)


@rank_zero_only
def print_section(section_name: str, end: bool = False):
    """print section name"""
    if end:
        print("#" * 10 + "#" + "#" * len(section_name) + "#" + "#" * 10)
    else:
        print("#" * 10 + " " + section_name + " " + "#" * 10)


def t2n(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor to a NumPy array."""
    assert isinstance(tensor, torch.Tensor), "Input must be a PyTorch tensor"
    return tensor.detach().cpu().numpy()


def n2t(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor."""
    type_check(array, np.ndarray)
    return torch.FloatTensor(array)


def check_tensor_blowup(t: torch.Tensor | np.ndarray, name: str = "tensor"):
    """
    Check if a tensor has NaN or Inf values and raise an error if so.

    Args:
        t (torch.Tensor | np.ndarray): Tensor to check.
        name (str): Optional name to include in the error message.
    """
    if isinstance(t, torch.Tensor):
        if not torch.isfinite(t).all():
            n_nan = torch.isnan(t).sum().item()
            n_inf = torch.isinf(t).sum().item()
            raise ValueError(
                f"{name} has blown up! "
                f"NaNs: {n_nan}, Infs: {n_inf}, "
                f"min={t.min().item()}, max={t.max().item()}"
            )
    elif isinstance(t, np.ndarray):
        if np.isnan(t).any() or np.isinf(t).any():
            n_nan = np.isnan(t).sum().item()
            n_inf = np.isinf(t).sum().item()
            raise ValueError(
                f"{name} has blown up! "
                f"NaNs: {n_nan}, Infs: {n_inf}, "
                f"min={t.min().item()}, max={t.max().item()}"
            )
    else:
        raise ValueError(f"Invalid input type: {type(t).__name__}")


def get_logger(name: str, group: str = None):
    """Get a logger with a unique version based on the current time."""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use WandbLogger for better integration with Weights & Biases
    logger = WandbLogger(
        name=f"experiment-{current_time}",  # this appears in wandb dashboard
        project=name,
        group=group,  # groups runs together in the W&B UI (e.g. ablation studies)
        log_model=False,
    )
    return logger
