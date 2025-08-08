# examples/oneDCorr/run_FLOREN.py
"""
Flow-matching Residual Embedded Neural Operator for One D problem
"""
import torch
import wandb
import argparse
import pytorch_lightning as L
import yaml
from omegaconf import OmegaConf
from mfFlow.utils import (
    printer,
    OpDataModule,
    get_checkpointer,
    get_trainer,
    init_weights,
    check_path,
)
from mfFlow.utils import OptimizedInference as Inference
from mfFlow.flow import Flow
from mfFlow.archs import get_embedding_modules

parser = argparse.ArgumentParser(description="Run oneDCorr with specified parameters.")
parser.add_argument(
    "--config",
    type=str,
    default="config_FLOREN.yml",
    help="Path to the configuration file.",
)

args = parser.parse_args()
config = OmegaConf.load(args.config)

torch.set_float32_matmul_precision("medium")  # for tensor cores


def print_header(config: dict):
    """Print the Header
    Args:
        config (dict): Configuration dictionary parameters.
    """
    printer("==" * 50)
    printer(f"Running oneDCorr with FLOREN ({config.job_name})")
    printer(f"Configuration file: {args.config}")
    printer(f"Tune hyperparameters: {config.tune_hyperparameters}")
    printer(f"Multi-fidelity Flow: {config.mfFlow}")
    printer(f"Number of samples: {config.data.high_fidelity.n_samples}")
    printer(f"Number of sensors: {config.data.high_fidelity.n_sensors}")
    printer("==" * 50)


def build_data_module(
    config: dict, hp_config: wandb.sdk.wandb_config.Config | dict = None
):
    """Get the data module for the oneDCorr problem.
    Args:
        config (dict): Configuration dictionary containing data parameters.
        hp_config (dict): Hyperparameter configuration dictionary.
    """
    # data loader config
    dataloader_config = dict(config.dataloader.copy())
    # set batch size from hyperparameter config
    dataloader_config["batch_size"] = hp_config["batch_size"]

    data_module = OpDataModule(
        nx=config.data.nx,
        nc=config.data.nc,
        nd=config.data.nd,
        low_fidelity_path=config.data.low_fidelity.path,
        high_fidelity_path=config.data.high_fidelity.path,
        n_samples=config.data.high_fidelity.n_samples,
        n_sensors=config.data.high_fidelity.n_sensors,
        mfFlow=config.mfFlow,
        sensing_strategy=config.data.sensing_strategy,
        dataloader_config=dataloader_config,
        test_data_path=config.data.test_data_path,
    )

    # Setup the data module
    data_module.setup()

    return data_module


def build_checkpointer(config: dict):
    """Get the checkpointer for saving and loading model checkpoints."""
    # get the checkpointer
    ckp_save_path = config.checkpoint_save_path
    checkpointer = get_checkpointer(
        ckp_save_path + "/mfFlow" if config.mfFlow else ckp_save_path
    )
    return checkpointer


def build_trainer(
    config: dict,
    hp_config: wandb.sdk.wandb_config.Config | dict = None,
    checkpointer=None,
):
    """Get the trainer for training the model.
    Args:
        config (dict): Configuration dictionary containing training parameters.
        hp_config (wandb.sdk.wandb_config.Config | dict): Hyperparameter configuration
        checkpointer: Checkpointer for saving and loading model checkpoints.
    Returns:
        trainer (pytorch_lightning.Trainer): Trainer object for training the model.
    """
    # logger name
    logger_name = config.logger_name.lower().strip()
    # train config
    train_config = dict(config.train.copy())
    train_config["max_epochs"] = hp_config["max_epochs"]

    trainer = get_trainer(
        checkpointer=checkpointer,
        logger_name=logger_name + "_mfFlow" if config.mfFlow else logger_name,
        train_config=train_config,
    )

    return trainer


class ResFlow(Flow, L.LightningModule):
    """Class for the residual flow model."""

    def __init__(
        self, config: dict, hp_config: wandb.sdk.wandb_config.Config | dict = None
    ):
        # Initialize the Flow and LightningModule
        Flow.__init__(self, hp_config=hp_config)
        L.LightningModule.__init__(self)

        # convert to dict for saving
        # config_dict = OmegaConf.to_container(config, resolve=True)
        hp_config_dict = dict(hp_config)

        # save the config and hyperparameter config
        self.save_hyperparameters({"config": config, "hp_config": hp_config_dict})

        self.config = config
        self.nx = self.config.data.nx
        self.nc = self.config.data.nc
        self.nd = self.config.data.nd

        # flow config
        self.flow_config = self.config.flow.copy()
        self.flow_config["time_embed_freq"] = hp_config["time_embed_freq"]
        self.flow_config["latent_dim"] = hp_config["latent_dim"]
        self.flow_config["num_centers"] = hp_config["num_centers"]

        self.sig_min = self.flow_config.sig_min
        self.time_embed_freq = self.flow_config.time_embed_freq
        self.latent_dim = self.flow_config.latent_dim
        self.num_centers = self.flow_config.num_centers

        # embedding modules
        embedding = get_embedding_modules(
            nx=self.nx,
            nc=self.nc,
            nd=self.nd,
            latent_dim=self.latent_dim,
            time_embed_freq=self.time_embed_freq,
            num_centers=self.num_centers,
        )
        self.state_embedding = embedding.get("state_embedding")
        self.condition_embedding = embedding.get("condition_embedding")
        self.fusion_embedding = embedding.get("fusion_embedding")
        self.domain_embedding = embedding.get("domain_embedding")

    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample from the base density"""
        return torch.randn_like(x1, device=x1.device)

    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """get the initial condition for the flow"""
        return torch.randn(batch_size, n_gen, self.nx, device=self.device)


def train_model(hp_config: dict = None):
    """Train the model using the specified configuration and hyperparameters.
    Returns:
        best_model_path (str): Path to the best model checkpoint after training.
    """
    printer("Training ...")
    if config.tune_hyperparameters:
        # initialize wandb with the hyperparameter config
        wandb.init()
        hp_config = wandb.config
    else:
        assert (
            hp_config is not None
        ), "Hyperparameter config must be provided for training."
    # data module
    data_module = build_data_module(config=config, hp_config=hp_config)
    # checkpointer
    checkpointer = build_checkpointer(config=config)
    # get trainer
    trainer = build_trainer(
        config=config, hp_config=hp_config, checkpointer=checkpointer
    )
    # model
    model = ResFlow(config=config, hp_config=hp_config)
    model.apply(init_weights)
    # load checkpoint if specified
    if config.checkpoint_load_path is not None:
        check_path(config.checkpoint_load_path)
        best_model_path = config.checkpoint_load_path
        printer(f"Loading checkpoint from {best_model_path}")
        # load the checkpoint
        model = ResFlow.load_from_checkpoint(best_model_path, map_location="cpu")
        model.to("cuda")
    # train
    printer("Starting training...")
    trainer.fit(model, data_module)

    if config.tune_hyperparameters:
        # clean up wandb run
        wandb.finish()
    else:
        # best model path
        best_model_path = checkpointer.best_model_path
        printer(f"Best model saved at {best_model_path}")
        return best_model_path, data_module


def infer_model(best_model_path, data_module):
    """Infererence task"""
    printer("Inference...")
    # load the best model
    best_model = ResFlow.load_from_checkpoint(
        best_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    # set model to eval mode
    best_model.eval()
    # enable inference model optimizations
    if hasattr(best_model, "compile") and torch.cuda.is_available():
        # Use torch.compile for PyTorch 2.0+ (significant speedup)
        printer("Compiling the model...")
        best_model = torch.compile(best_model, mode="max-autotune")

    # infer the mode
    infer = Inference(
        model=best_model,
        test_config=data_module.test_config,
        statistics=data_module.statistics,
        job_name=config.job_name,
        mfFlow=config.mfFlow,
        generate_config=config.generate,
    )

    infer()


if __name__ == "__main__":
    # print header
    print_header(config)
    # if tune hyperparameters is True, load the hyperparameter config
    if config.tune_hyperparameters:
        # load the hyperparameter config from a yaml file
        with open("config_sweep.yml", "r") as file:
            hp_config = yaml.safe_load(file)
        # initialize agent
        sweep_id = wandb.sweep(hp_config)
        wandb.agent(sweep_id, function=train_model, count=100)
    else:
        # load the hyperparameter config
        with open("config_hyperparameters.yml", "r") as file:
            hp_config = yaml.safe_load(file)["FLOREN"]
        # get the best model path (training or evaluation)
        if config.train.stage == "train":
            best_model_path, data_module = train_model(hp_config)
        elif config.train.stage == "eval":
            printer(
                "Skipping training and evaluating directly using"
                f"{config.checkpoint_load_path}"
            )
            assert (
                config.dataloader.reload is True
            ), "Reload must be True for evaluation"
            assert (
                config.checkpoint_load_path is not None
            ), "checkpoint must be provided in eval mode"
            # data module
            data_module = build_data_module(config=config, hp_config=hp_config)
            # load the checkpoint
            check_path(config.checkpoint_load_path)
            best_model_path = config.checkpoint_load_path
        # infer
        infer_model(best_model_path, data_module)
