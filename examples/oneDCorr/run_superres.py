import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from floral.gp import get_gp_prior
from floral.archs import get_vector_field_operator
import argparse
import torch.nn as nn
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        prog="onedcorr",
        description="onedcorr superresolution",
    )

    parser.add_argument(
        "-nT",
        "--n_train",
        type=int,
        default=10,
        help="Number of train samples",
    )

    parser.add_argument(
        "--train_res",
        type=int,
        default=64,
        help="Training resolution",
    )

    parser.add_argument(
        "-nV",
        "--n_val",
        type=int,
        default=10,
        help="Number of val samples",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of train samples",
    )

    parser.add_argument("--floral", action="store_true")

    args = parser.parse_args()
    return args


def seed_everything(seed: int = 42):
    """seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(args):
    """dataloader"""
    data_LF = torch.load("low_fidelity.pt", weights_only=False)
    data_HF = torch.load("high_fidelity.pt", weights_only=False)

    condition = data_HF.get("condition")
    target_field = data_HF.get("field")
    if args.floral:
        target_field -= data_LF.get("field")
    # check resolution
    assert target_field.shape[-1] % args.train_res == 0
    # split
    n_samples = len(target_field)
    assert args.n_train + args.n_val <= n_samples
    train_field, train_condition = (
        target_field[: args.n_train],
        condition[: args.n_train],
    )
    val_field, val_condition = target_field[-args.n_val :], condition[-args.n_val :]
    train_LF_field, val_LF_field = (
        data_LF.get("field")[: args.n_train],
        data_LF.get("field")[-args.n_val :],
    )
    # normalize
    field_mean = train_field.mean(dim=(0, *range(2, train_field.ndim)), keepdim=True)
    field_std = train_field.std(dim=(0, *range(2, train_field.ndim)), keepdim=True)

    condition_mean = train_condition.mean(
        dim=(0, *range(2, train_condition.ndim)), keepdim=True
    )
    condition_std = train_condition.std(
        dim=(0, *range(2, train_condition.ndim)), keepdim=True
    )

    train_field = (train_field - field_mean) / field_std
    val_field = (val_field - field_mean) / field_std

    train_condition = (train_condition - condition_mean) / condition_std
    val_condition = (val_condition - condition_mean) / condition_std

    full_domain = data_HF.get("field_domain").T.unsqueeze(0)

    train_set = TensorDataset(train_field, train_condition, train_LF_field)
    val_set = TensorDataset(val_field, val_condition, val_LF_field)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    stats = {
        "field": {"mean": field_mean, "std": field_std},
        "condition": {"mean": condition_mean, "std": condition_std},
    }

    return train_loader, val_loader, full_domain, stats


class Flow(nn.Module):
    def __init__(self, floral, device):
        super(Flow, self).__init__()
        self.sig_min = 1e-5
        self.floral = floral
        self.device = device

        # operator config
        self.operator_config = self._get_operator_config()
        self.vector_field = get_vector_field_operator(
            operator_config=self.operator_config, floral=self.floral
        )

        # prior
        prior_scale = 2.0 if self.floral else 1.0
        lengthscale = 1e-3
        outputscale = 1.0
        self.prior = get_gp_prior(
            lengthscale=lengthscale,
            outputscale=outputscale * prior_scale,
            confidence=0.0,  # no bias
        )
        # noise
        self.noise = get_gp_prior(
            lengthscale=lengthscale,
            outputscale=outputscale,
            confidence=0.0,  # no bias
        )

    def _get_operator_config(self):
        field_config = {
            "channels": 1,
            "ndim": 1,
            "hidden_channels": 64,
            "lifting_channel_ratio": 4,
            "projection_channel_ratio": 4,
            "n_layers": 4,
            "modes": 64,
        }

        condition_config = {
            "channels": 1,
            "ndim": 1,
        }

        operator_config = {
            "method": "FiLMFNO",
            "field": field_config,
            "condition": condition_config,
        }
        return operator_config

    def _sample_prior_measure(self, batch_size, LF_field, domain):
        domain_eval = domain.flatten(2).transpose(1, 2).squeeze(0)
        field_channels = self.operator_config["field"]["channels"]
        field_dims = list(LF_field.shape[2:])
        # sample prior
        prior_samples = self.prior.sample(
            domain=domain_eval,
            batch_size=batch_size,
            field_channels=field_channels,
            field_dims=field_dims,
        )
        return prior_samples

    def _sample_noise_measure(self, batch_size, domain):
        domain_eval = domain.flatten(2).transpose(1, 2).squeeze(0)
        field_channels = self.operator_config["field"]["channels"]
        field_dims = list(domain.shape[2:])
        noise_samples = self.noise.sample(
            domain=domain_eval,
            batch_size=batch_size,
            field_channels=field_channels,
            field_dims=field_dims,
        )
        return noise_samples

    def _sample_conditional_flow(self, target_field, prior, t, domain):
        """sample conditional flow"""
        B = len(target_field)
        field_ndim = self.operator_config["field"]["ndim"]
        # reshape t
        t_expand = t.view(B, 1, *([1] * field_ndim))
        # sample noise
        noise_scale = torch.mean(
            (target_field - prior) ** 2,
            dim=list(range(1, target_field.ndim)),
            keepdim=True,
        )
        noise = self._sample_noise_measure(B, domain)
        noise = self.sig_min * noise_scale * noise
        # sample from conditional path
        psi = (t_expand * target_field + (1.0 - target_field) * prior) + noise
        return psi

    def _comp_conditional_flow_derivative(self, target_field, prior):
        return target_field - prior

    def comp_loss(self, target_field, condition, LF_field, domain):
        B, C, *dims = target_field.shape
        # time samples
        t = torch.rand(B, 1, device=self.device)
        # prior
        prior = self._sample_prior_measure(B, LF_field, domain)
        # conditional path samples
        psi = self._sample_conditional_flow(target_field, prior, t, domain)
        # target field
        psi_prime = self._comp_conditional_flow_derivative(
            target_field=target_field, prior=prior
        )
        # model field
        vt = self.vector_field(
            psi=psi,
            condition=condition,
            LF_field=LF_field,
            field_domain=domain,
            t=t,
        )
        # weighted loss
        w_t = 1.0 + 2.0 * t**2
        loss_raw = ((vt - psi_prime) ** 2).mean(dim=list(range(1, vt.ndim)))
        loss = (w_t.squeeze() * loss_raw).mean()
        return loss


class Monitor:
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self._reset()

    def _reset(self):
        self.current_val = None
        self.current_avg_val = None

    def update(self, val):

        if self.current_avg_val is None:
            self.current_avg_val = val
        else:
            self.current_avg_val = (
                self.gamma * self.current_val + (1.0 - self.gamma) * val
            )

        self.current_val = val


def train(
    flow,
    train_loader,
    val_loader,
    full_domain,
    train_res: int = 64,
    device="cpu",
    epochs=300,
):
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e3, weight_decay=1e-4)
    pbar = tqdm(range(epochs), desc=f"Training (Res: {train_res})")
    for ii in pbar:
        # train
        flow.train()
        for target_field, condition, LF_field in train_loader:
            optimizer.zero_grad()
            # slice
            skip = target_field.shape[-1] // train_res
            slice_fields = (slice(None),) * 2 + (
                slice(0, target_field.shape[-1], skip),
            )
            target_field = target_field[slice_fields].to(device)
            condition = condition[slice_fields].to(device)
            LF_field = LF_field[slice_fields].to(device)
            domain = full_domain[slice_fields].to(device)
            # compute loss
            loss = flow.comp_loss(
                target_field=target_field,
                condition=condition,
                LF_field=LF_field,
                domain=domain,
            )
            # optimize
            loss.backward()
            optimizer.step()

        # val
        flow.eval()
        with torch.no_grad():
            for target_field, condition, LF_field in val_loader:
                # slice
                skip = target_field.shape[-1] // train_res
                slice_fields = (slice(None),) * 2 + (
                    slice(0, target_field.shape[-1], skip),
                )
                target_field = target_field[slice_fields].to(device)
                condition = condition[slice_fields].to(device)
                LF_field = LF_field[slice_fields].to(device)
                domain = full_domain[slice_fields].to(device)
                # compute loss
                loss = flow.comp_loss(
                    target_field=target_field,
                    condition=condition,
                    LF_field=LF_field,
                    domain=domain,
                )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")
    seed_everything()
    args = parse_args()
    train_loader, val_loader, full_domain, stats = get_dataloader(args)
    flow = Flow(floral=args.floral, device=device).to(device)
    train(
        flow=flow,
        train_loader=train_loader,
        val_loader=val_loader,
        full_domain=full_domain,
        device=device,
        train_res=args.train_res,
        epochs=300,
    )
