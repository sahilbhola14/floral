# examples/oneDCorr/run_floral.py
"""
Flow-matching operator for residual-augmented learning for 1D toy problem
"""
import torch
import wandb
import argparse
import lightning as L
import yaml
from omegaconf import OmegaConf
from floral.utils import (
    printer,
    OpDataModule,
    get_checkpointer,
    get_trainer,
    init_weights,
    check_path,
)
from floral.utils import Inference
from floral.flow import Flow

parser = argparse.ArgumentParser(description="Run oneDCorr with specified parameters.")
parser.add_argument(
    "--config",
    type=str,
    default="config_floral.yml",
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
    printer("Running oneDCorr")
    printer(f"Job name: {config.job_name}")
    printer(f"Configuration file: {args.config}")
    printer(f"Tune hyperparameters: {config.tune_hyperparameters}")
    printer(f"Multi-fidelity Flow: {config.floral}")
    printer(f"Number of samples: {config.data.n_samples}")
    printer("==" * 50)


def build_data_module(
    config: dict,
    hp_config: wandb.sdk.wandb_config.Config | dict = None,
    verbose: bool = False,
):
    """Get the data module for the oneDCorr problem.
    Args:
        config (dict): Configuration dictionary containing data parameters.
        hp_config (dict): Hyperparameter configuration dictionary.
    """
    # data loader config
    dataloader_config = dict(config.dataloader.copy())
    # set batch size from hyperparameter config
    dataloader_config["batch_size"] = hp_config.get("batch_size", 64)

    data_module = OpDataModule(
        low_fidelity_path=config.data.low_fidelity.path,
        high_fidelity_path=config.data.high_fidelity.path,
        n_samples=config.data.n_samples,
        dataloader_config=dataloader_config,
        floral=config.floral,
        verbose=verbose,
    )

    # Setup the data module
    data_module.setup()

    return data_module


def build_checkpointer(config: dict):
    """Get the checkpointer for saving and loading model checkpoints."""
    # get the checkpointer
    ckp_save_path = config.checkpoint_save_path
    checkpointer = get_checkpointer(
        ckp_save_path + "/floral" if config.floral else ckp_save_path
    )
    return checkpointer


def build_trainer(
    config: dict,
    hp_config: wandb.sdk.wandb_config.Config | dict = None,
    checkpointer=None,
    verbose: bool = False,
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
    train_config["max_epochs"] = hp_config.get("max_epochs", 100)

    trainer = get_trainer(
        checkpointer=checkpointer,
        logger_name=logger_name + "_floral" if config.floral else logger_name,
        train_config=train_config,
        verbose=verbose,
    )

    return trainer


class ResFlow(L.LightningModule):
    """Class for the residual flow model."""

    def __init__(
        self,
        config: dict,
        hp_config: wandb.sdk.wandb_config.Config | dict,
        domains: dict,
    ):
        super(ResFlow, self).__init__()

        # normalize domains
        field_domain = domains.get("field", None)
        condition_domain = domains.get("condition", None)
        if field_domain is not None and not torch.is_tensor(field_domain):
            field_domain = torch.tensor(field_domain)
        if condition_domain is not None and not torch.is_tensor(condition_domain):
            condition_domain = torch.tensor(condition_domain)
        domains = {"field": field_domain, "condition": condition_domain}

        # initialize the flow
        self.flow = Flow(config=config, hp_config=hp_config, domains=domains)

        # convert to dict for saving
        hp_config_dict = dict(hp_config)

        # save the config and hyperparameter config
        self.save_hyperparameters(
            {
                "config": config,
                "hp_config": hp_config_dict,
                "domains": {
                    "field": field_domain.tolist(),
                    "condition": condition_domain.tolist(),
                },
            }
        )

    def configure_optimizers(self, verbose=False):
        return self.flow.configure_optimizers(verbose=verbose)

    def training_step(self, batch, batch_idx):
        """training step"""
        loss = self.flow.training_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""
        loss = self.flow.validation_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def load_state_dict(self, state_dict, strict=True):
        # drop PyTorch-internal metadata if present
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)


def train_model(hp_config: dict = None):
    """Train the model using the specified configuration and hyperparameters.
    Returns:
        best_model_path (str): Path to the best model checkpoint after training.
    """
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
    model = ResFlow(
        config=config,  # general config
        hp_config=hp_config,  # hyperparameter config
        domains=data_module.domains,  # field and condition domains
    )
    # if hasattr(model, "compile") and torch.cuda.is_available():
    #     printer("Compiling the model...")
    #     model = torch.compile(model, mode="default")
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
    best_model = ResFlow.load_from_checkpoint(best_model_path)
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
        val_set=data_module.val_set,
        domains=data_module.domains,
        statistics=data_module.statistics,
        job_name=config.job_name,
        floral=config.floral,
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
            hp_config = yaml.safe_load(file)
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
