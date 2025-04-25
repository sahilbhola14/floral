import os
import os.path as osp
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

import pytorch_lightning as L
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 14
plt.rcParams["lines.linewidth"] = 2


@rank_zero_only
def printer(message):
    print(message)


def t2n(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().numpy()


def n2t(array: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(array)


def make_dirs(dirname) -> None:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def check_path(path: str) -> None:
    assert osp.exists(path), f"Path {path} does not exist"


def get_checkpointer(path: str):
    """checkpointer"""
    ckp = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    return ckp


def restrict_domain(
    x: torch.Tensor, source_domain: torch.Tensor, target_domain: torch.Tensor
):
    """Function evaluates the tensor x at the points in source_domain and returns
    the values at the points in target_domain.
    Args:
    x: torch.Tensor(batch_size, len(source_domain))
    source_domain: torch.Tensor(len(source_domain))
    target_domain: torch.Tensor(len(target_domain))
    Returns:
    torch.Tensor(batch_size, len(target_domain))
    """

    batch_size, n_D = x.shape

    # Normalize the source and target domains to [-1, 1]
    x_d_norm = (
        2
        * (target_domain - source_domain.min())
        / (source_domain.max() - source_domain.min())
        - 1
    )

    # Reshape data to match grid_sample input requirements
    data = x.unsqueeze(1)  # (batch_size, 1, n_D)
    x_d_norm = x_d_norm.view(1, -1, 1).expand(
        batch_size, -1, -1
    )  # (batch_size, n_d, 1)

    # Add a dummy second coordinate (0) to make it (batch_size, n_d, 2)
    x_d_norm = torch.cat(
        [x_d_norm, torch.zeros_like(x_d_norm)], dim=-1
    )  # (batch_size, n_d, 2)

    # Perform interpolation
    interpolated = (
        F.grid_sample(
            data.unsqueeze(1),
            x_d_norm.unsqueeze(1),
            align_corners=True,
            mode="bilinear",
        )
        .squeeze(1)
        .squeeze(1)
    )

    assert interpolated.shape == (batch_size, len(target_domain))

    return interpolated


def get_logger(name: str):
    """logger"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger("logs", name=name, version=f"{current_time}")
    return logger


def get_trainer(checkpointer, logger_name: str, train_config: dict):
    # get the logger
    logger = get_logger(logger_name)
    # trainer
    trainer = L.Trainer(
        logger=logger,
        max_epochs=train_config.max_epochs,
        devices=train_config.devices,
        accelerator=train_config.accelerator,
        strategy=train_config.strategy,
        callbacks=[checkpointer],
    )

    return trainer


def get_path(path: list) -> str:
    return os.path.join(*path)


def save_args(args, path: str) -> None:
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved args to {path}")


def save_checkpoint(state, path: str) -> None:
    torch.save(state, path)


def plot4(
    x1_train,
    x0_train_hat,
    x1_hat,
    x0_hat,
    log_density_x1_hat,
    log_density_x1,
    denormalizer,
    visualize=False,
    save=False,
):
    ndims = x1_train.shape[1]
    assert ndims <= 2, "Can only plot 2D data"

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    x1_train = denormalizer(x=x1_train)  # denormalize the data
    x1_hat = denormalizer(x=x1_hat)  # denormalize the data

    if ndims == 1:

        axs[0, 0].hist(x1_train.flatten(), bins=100, alpha=0.5, density=True)
        if log_density_x1 is not None:
            axs[0, 0].scatter(
                x1_train.flatten(), torch.exp(log_density_x1).flatten(), s=1, c="r"
            )
        axs[0, 0].set_title("Training data samples")

        axs[0, 1].hist(x0_train_hat.flatten(), bins=100, alpha=0.5, density=True)
        axs[0, 1].set_title("Encoded training data samples")

        axs[1, 0].hist(x1_hat.flatten(), bins=100, alpha=0.5, density=True)
        if log_density_x1_hat is not None:
            axs[1, 0].scatter(
                x1_hat.flatten(), torch.exp(log_density_x1_hat).flatten(), s=1, c="r"
            )
        axs[1, 0].set_title("Generated data samples")

        axs[1, 1].hist(x0_hat.flatten(), bins=100, alpha=0.5, density=True)
        axs[1, 1].set_title("Encoded generated data samples")

    elif ndims == 2:
        nBins = 33
        LOWX = -4
        HIGHX = 4
        LOWY = -4
        HIGHY = 4

        im1, _, _, map1 = axs[0, 0].hist2d(
            x1_train[:, 0],
            x1_train[:, 1],
            range=[[LOWX, HIGHX], [LOWY, HIGHY]],
            bins=nBins,
        )
        axs[0, 0].set_title("Training data samples")

        im2, _, _, map2 = axs[0, 1].hist2d(
            x0_train_hat[:, 0],
            x0_train_hat[:, 1],
            range=[[-4, 4], [-4, 4]],
            bins=nBins,
        )
        axs[0, 1].set_title("Encoded training data samples")

        im3, _, _, map3 = axs[1, 0].hist2d(
            x1_hat[:, 0],
            x1_hat[:, 1],
            range=[[LOWX, HIGHX], [LOWY, HIGHY]],
            bins=nBins,
        )
        axs[1, 0].set_title("Generated data samples")
        if log_density_x1_hat is not None:
            sc = axs[1, 1].scatter(
                x1_hat[:, 0],
                x1_hat[:, 1],
                c=torch.exp(log_density_x1_hat).flatten(),
                s=2,
                edgecolor=None,
            )
            axs[1, 1].set_xlim(LOWX, HIGHX)
            axs[1, 1].set_ylim(LOWY, HIGHY)
            plt.colorbar(sc, ax=axs[1, 1], orientation="vertical")
            axs[1, 1].set_title("Probability density of generated data samples")

    if visualize:
        plt.tight_layout()
        if save:
            plt.savefig("plot4.png", dpi=300)
        plt.show()
    if save:
        plt.savefig("plot4.png", dpi=300)
