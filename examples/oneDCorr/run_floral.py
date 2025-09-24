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
from floral.utils import OptimizedInference as Inference
from floral.flow import Flow
from floral.archs import get_operator_modules
from floral.GP import GPPrior

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
    dataloader_config["batch_size"] = hp_config["batch_size"]

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
    train_config["max_epochs"] = hp_config["max_epochs"]

    trainer = get_trainer(
        checkpointer=checkpointer,
        logger_name=logger_name + "_floral" if config.floral else logger_name,
        train_config=train_config,
        verbose=verbose,
    )

    return trainer


class ResFlow(Flow, L.LightningModule):
    """Class for the residual flow model."""

    def __init__(
        self,
        config: dict,
        hp_config: wandb.sdk.wandb_config.Config | dict = None,
        domains: dict = None,
    ):
        # create flow config
        flow_config = config.flow.copy()
        flow_config["time_embed_freq"] = hp_config.get("time_embed_freq", 4)

        # initialize the Flow and LightningModule
        Flow.__init__(
            self, hp_config=hp_config, domains=domains, flow_config=flow_config
        )
        L.LightningModule.__init__(self)

        # convert to dict for saving
        hp_config_dict = dict(hp_config)

        # save the config and hyperparameter config
        self.save_hyperparameters(
            {
                "config": config,
                "hp_config": hp_config_dict,
            }
        )

        self.config = config

        # operator params
        self.operator_params = self._get_operator_params()

        # operator modules
        operator_modules = get_operator_modules(
            operator_params=self.operator_params, time_embed_dim=self.time_embed_dim
        )

        # extract operator modules
        self.field_operator = operator_modules.get("field_operator")
        self.condition_operator = operator_modules.get("condition_operator")

        # prior
        self.prior = GPPrior()

    def _get_operator_params(self):
        """extract the operator params"""
        dims = {
            "field": self.config.data.shape.field.get("dims", 1),
            "condition": self.config.data.shape.condition.get("dims", 1),
        }
        ch_in = {
            "field": self.config.data.shape.field.get("ch_in", 1),
            "condition": self.config.data.shape.condition.get("ch_in", 1),
        }
        ch_out = {
            "field": self.config.data.shape.field.get("ch_in", 1),  # same as ch_in
            "condition": self.config.flow.operator.condition.get("ch_out", 1),
        }
        ch_hidden = {
            "field": self.config.flow.operator.field.get("ch_hidden", 32),
            "condition": self.config.flow.operator.condition.get("ch_hidden", 32),
        }
        n_modes = {
            "field": self.config.flow.operator.field.get("field", 32),
            "condition": self.config.flow.operator.condition.get("condition", 32),
        }

        operator_params = {
            "dims": dims,
            "ch_in": ch_in,
            "ch_out": ch_out,
            "ch_hidden": ch_hidden,
            "n_modes": n_modes,
        }

        return operator_params

    def _sample_base_measure(
        self,
        field_domain: torch.Tensor,
        field_grid: tuple,
        field_ch: int,
        n_samples: int,
    ):
        # flatten domain
        domain_flat = field_domain.squeeze(0).flatten(start_dim=1).T
        # sample from prior
        prior_samples = self.prior.sample(
            domain=domain_flat,
            grid=field_grid,
            n_channels=field_ch,
            n_samples=n_samples,
        )
        # check shape
        B, ch, *grid = prior_samples.shape
        dim_flag = all([ii == jj for ii, jj in zip(grid, field_grid)])
        assert B == n_samples, f"Batch mismatch: {B} vs {n_samples}"
        assert ch == field_ch, f"Channel mismatch: {ch} vs {field_ch}"
        assert dim_flag, f"Dimension mismatch: {grid} vs {field_grid}"

        return prior_samples


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
    raise NotImplementedError("Inference for oneDCorr is not implemented yet.")
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
