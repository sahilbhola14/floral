"""
Test the training of flow matching on the residual
Notes:
    1. increasing sigma_noise reduces the correlation of the residual with the
    low-fidelty model. Emperically, it was observed that better the correlation,
    better the model performance.
    2. Making the vector field model conditioned on the low-fidelity helps improves the
    mean and reduces the variance.
    3. If learning the residual, the base sample comes from r_0 = x_0 - x_0_hat. In case
    x_0 and x_0_hat are N(0, 1) distributed, r_0 is N(0, 2) distributed since
    x_0 and x_0_hat are not correlated.
"""

import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchdiffeq import odeint

SIGMA_NOISE = 1.2


def seed_everything(seed: int = 42):
    """seed everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_high_fidelity(n_samples, cond=None):
    """sample high-fidelity"""

    def _func(cond):
        assert cond.ndim == 2 and cond.shape[-1] == 1
        return cond + torch.randn(len(cond), 1)

    if cond is None:
        p = torch.rand(n_samples)
        mean = torch.ones(n_samples) * 5
        mean[p < 0.5] = -5
        mean = mean.view(-1, 1)
        return _func(mean), mean
    else:
        return _func(cond), cond


def sample_low_fidelity(x_hf):
    # increasing this reduces the correlation of the residual with the low-fidelity.
    # return 0.8*x_hf + 0.6 * torch.randn_like(x_hf)
    return 0.6 * x_hf + SIGMA_NOISE * torch.randn_like(x_hf)


def plot(x_hf, x_lf, savepath="data.png"):
    # flatten
    xh = x_hf.ravel()
    xl = x_lf.ravel()

    # correlation
    r, _ = pearsonr(xh, xl)
    r_res, _ = pearsonr(xh - xl, xl)

    fig, axs = plt.subplots(1, 4, figsize=(10, 3), layout="compressed")

    # 1. state marginals
    axs[0].hist(xh, density=True, bins=50, alpha=0.7, label="HF")
    axs[0].hist(xl, density=True, bins=50, alpha=0.7, label="LF")
    axs[0].set_title("Marginals")
    axs[0].legend()

    # 2. residual marginal
    axs[1].hist((xh - xl), density=True, bins=50)
    axs[1].set_title("Residual (HF - LF)")

    # 3. state scatter
    axs[2].scatter(xl, xh, s=3, alpha=0.5)
    axs[2].set_xlabel("Low-fidelity")
    axs[2].set_ylabel("High-fidelity")
    axs[2].set_title("HF vs LF")
    minv = min(xl.min(), xh.min())
    maxv = max(xl.max(), xh.max())
    axs[2].plot([minv, maxv], [minv, maxv], "r--", lw=1)
    axs[2].set_title(f"Correlation: {r: .4f}")

    # 4. residual scatter
    axs[3].scatter(xh - xl, xl, s=3, alpha=0.5)
    axs[3].set_xlabel("Low-fidelity")
    axs[3].set_ylabel("Residual")
    axs[3].set_title("Residual vs LF")
    minv = min(xl.min(), (xh - xl).min())
    maxv = max(xl.max(), (xh - xl).max())
    axs[3].plot([minv, maxv], [minv, maxv], "r--", lw=1)
    axs[3].set_title(f"Correlation: {r_res: .4f}")

    plt.savefig(savepath, dpi=150)
    plt.close()


def get_dataloaders(n_samples=10000, batch_size: int = 128):
    x_hf, mean = sample_high_fidelity(n_samples)
    x_lf = sample_low_fidelity(x_hf)
    plot(x_hf=x_hf, x_lf=x_lf)
    n_train = int(0.7 * n_samples)

    x_hf_train, x_hf_val = x_hf[:n_train], x_hf[n_train:]
    x_lf_train, x_lf_val = x_lf[:n_train], x_lf[n_train:]
    mean_train, mean_val = mean[:n_train], mean[n_train:]

    train_set = TensorDataset(x_hf_train, x_lf_train, mean_train)
    val_set = TensorDataset(x_hf_val, x_lf_val, mean_val)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def wasserstein_2(x1_gen, x1_hf):
    """
    Compute 1D Wasserstein-2 distance between generated and HF samples.
    Returns W2 (not squared).
    """
    xg = torch.sort(x1_gen.flatten())[0]
    xh = torch.sort(x1_hf.flatten())[0]
    return torch.sqrt(torch.mean((xg - xh) ** 2))


def _gaussian_kde_logpdf(samples, points, bandwidth):
    """
    Compute log p(points) under KDE built from samples.
    samples: (N,)
    points:  (M,)
    """
    diff = points[:, None] - samples[None, :]
    kernel_vals = torch.exp(-0.5 * (diff / bandwidth) ** 2)
    pdf = kernel_vals.mean(dim=1) / (bandwidth * math.sqrt(2 * math.pi))
    return torch.log(pdf + 1e-12)


def kl_divergence_kde(x1_gen, x1_hf):
    """
    KL(p_gen || p_hf) using Gaussian KDE for continuous distributions.
    """
    xg = x1_gen.flatten()
    xh = x1_hf.flatten()

    # Silverman bandwidth
    def bandwidth(s):
        n = len(s)
        return 1.06 * s.std() * n ** (-1 / 5)

    bw_g = bandwidth(xg)
    bw_h = bandwidth(xh)

    log_p = _gaussian_kde_logpdf(xg, xg, bw_g)
    log_q = _gaussian_kde_logpdf(xh, xg, bw_h)  # evaluate q at samples of p

    return torch.mean(log_p - log_q)


class VectorField1D(nn.Module):
    """
    Neural network vector field for 1D problem.

    Takes (psi_t, t) as input and outputs the vector field value.
    Both inputs have shape (B, 1).
    """

    def __init__(
        self,
        state_dim=1,
        t_embed_freq=4,
        hidden_dim=64,
        num_layers=3,
        cond_dim=1,
        floral: bool = False,
    ):
        """
        Args:
            hidden_dim: Number of hidden units in each layer
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.t_embed_freq = t_embed_freq
        self.floral = floral

        # Input layer: concatenate psi_t and t
        in_feat = state_dim + 2 * t_embed_freq + cond_dim
        if self.floral:
            in_feat += state_dim
        layers = [nn.Linear(in_feat, hidden_dim), nn.SiLU()]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        # Output layer: single scalar output
        layers.append(nn.Linear(hidden_dim, state_dim))

        self.net = nn.Sequential(*layers)

    def _embed_time(self, t):
        """embedd the time"""
        freq = torch.arange(self.t_embed_freq, device=t.device).view(1, -1)
        freq = 2 * math.pi * freq
        return torch.cat(((t * freq).sin(), (t * freq).cos()), dim=-1)

    def forward(self, psi_t, t, cond=None, x1_hat=None):
        """
        Forward pass.

        Args:
            psi_t: Tensor of shape (B, 1), state at time t
            t: Tensor of shape (B, 1), time values

        Returns:
            Tensor of shape (B, 1), vector field values
        """
        # embed the time
        t_embed = self._embed_time(t)
        # Concatenate inputs
        x = torch.cat([psi_t, t_embed, cond], dim=1)
        if self.floral:
            x = torch.cat([psi_t, t_embed, cond, x1_hat], dim=1)
        else:
            x = torch.cat([psi_t, t_embed, cond], dim=1)

        # Forward through network
        out = self.net(x)
        return out


class FlowMatching(nn.Module):
    def __init__(self, floral: bool = False, device="cpu"):
        super(FlowMatching, self).__init__()
        self.sig_min = 1e-5
        self.floral = floral
        self.device = device
        self.model = VectorField1D(
            state_dim=1, t_embed_freq=4, cond_dim=1, floral=floral
        )

    def _sample_base_distribution(self, n_samples):
        """sample the base distribution"""
        if self.floral:
            return torch.randn(n_samples, 1, device=self.device) * math.sqrt(2)
        else:
            return torch.randn(n_samples, 1, device=self.device)

    def _sample_conditional_path(
        self, t: torch.Tensor, x1: torch.Tensor, x1_hat: torch.Tensor, r0: torch.Tensor
    ):
        """sample the conditional path"""
        B = len(x1)
        # compute the target
        r1 = (x1 - x1_hat) if self.floral else x1
        # sample noise
        noise = self.sig_min * torch.randn(B, 1, device=self.device)
        # sample conditional path
        psi_t = (t * r1 + (1.0 - t) * r0) + noise
        return psi_t

    def _evaluate_conditional_target_vector_field(
        self, x1: torch.Tensor, x1_hat: torch.Tensor, r0: torch.Tensor
    ):
        """evalute the conditional target vector field"""
        # compute the target
        r1 = (x1 - x1_hat) if self.floral else x1
        # conditional vector field
        u_t = r1 - r0
        return u_t

    def comp_loss(self, x1, x1_hat, cond=None):
        B = len(x1)
        # sample time from U[0, 1]
        t = torch.rand(B, 1, device=self.device)
        # sample base distribution
        r0 = self._sample_base_distribution(B)
        # sample conditional path
        psi_t = self._sample_conditional_path(t=t, x1=x1, x1_hat=x1_hat, r0=r0)
        # evaluate conditional vector field
        u_t = self._evaluate_conditional_target_vector_field(
            x1=x1, x1_hat=x1_hat, r0=r0
        )
        # evalute model vector field
        v_t = self.model(psi_t=psi_t, t=t, cond=cond, x1_hat=x1_hat)
        return torch.mean((v_t - u_t) ** 2)

    def rhs(self, t, xt, cond, x1_hat):
        t_eval = torch.ones(len(xt), 1, device=self.device) * t
        v_t = self.model(psi_t=xt, t=t_eval, cond=cond, x1_hat=x1_hat)
        return v_t

    @torch.no_grad()
    def generate(self, n_samples):
        """generate samples"""
        r0 = self._sample_base_distribution(n_samples)
        t = torch.linspace(0, 1, 100, device=self.device)
        cond = torch.ones(n_samples, 1, device=self.device) * 5

        x1_hf, _ = sample_high_fidelity(n_samples, cond=cond.to("cpu"))
        x1_lf = sample_low_fidelity(x1_hf).to(self.device)

        def _wrapper(t, xt):
            return self.rhs(t, xt, cond, x1_lf)

        r1_gen = odeint(_wrapper, r0, t, method="dopri5", atol=1e-5, rtol=1e-5)[-1].to(
            "cpu"
        )
        x1_lf = x1_lf.to("cpu")
        x1_gen = (x1_lf + r1_gen) if self.floral else r1_gen

        # distances
        w2_distance = wasserstein_2(x1_gen=x1_gen, x1_hf=x1_hf)
        kl_distance = kl_divergence_kde(x1_gen=x1_gen, x1_hf=x1_hf)
        header = "[FLORAL]" if self.floral else "[FLORA]"
        print(
            f"{header} "
            f"W2 distance: {w2_distance: .4f} | KL distance: {kl_distance: .4f}"
        )

        fig, axs = plt.subplots(1, 1, figsize=(4, 2), layout="compressed")
        axs.hist(x1_gen.ravel(), bins=100, density=True, label="Generated")
        axs.hist(x1_hf.ravel(), bins=100, density=True, label="HF")
        axs.hist(x1_lf.ravel(), bins=100, density=True, label="LF", alpha=0.3)
        axs.set_xlim(-5, 15)
        axs.set_ylim(0, 0.5)
        axs.set_title(
            f"W2 distance: {w2_distance: .4f} | KL distance: {kl_distance: .4f}"
        )
        axs.legend()
        plt.savefig("gen_floral.png" if self.floral else "gen_flora.png")


def train(flow, train_loader, val_loader, epochs=5000, device="cpu"):
    pbar = tqdm(range(epochs), desc="Training")
    optim = torch.optim.Adam(flow.parameters(), lr=1e-3)
    loss_train_list = []
    loss_val_list = []
    for ii in pbar:
        flow.train()
        loss_train = 0.0
        for x1, x1_hat, cond in train_loader:
            # move to device
            x1 = x1.to(device)
            x1_hat = x1_hat.to(device)
            cond = cond.to(device)
            # zero grad
            optim.zero_grad()
            # comp loss
            loss = flow.comp_loss(x1=x1, x1_hat=x1_hat, cond=cond)
            # step
            loss.backward()
            optim.step()
            loss_train += loss.item()
        loss_train = loss_train / len(train_loader)
        loss_train_list.append(loss_train)

        flow.eval()
        loss_val = 0.0
        with torch.no_grad():
            for x1, x1_hat, cond in val_loader:
                # move to device
                x1 = x1.to(device)
                x1_hat = x1_hat.to(device)
                cond = cond.to(device)
                # comp loss
                loss = flow.comp_loss(x1=x1, x1_hat=x1_hat, cond=cond)
                loss_val += loss.item()
            loss_val = loss_val / len(val_loader)
            loss_val_list.append(loss_val)

        pbar.set_postfix(
            {
                "train": f"{loss_train: .2e}",
                "val": f"{loss_val: .2e}",
            }
        )

    # generate samples
    flow.eval()
    flow.generate(50000)


def compare(train_loader, val_loader, epochs: int = 500, device="cpu"):
    print(f"running on {device}")
    # flora
    flow_flora = FlowMatching(device=device, floral=False).to(device)
    train(flow_flora, train_loader, val_loader, device=device, epochs=epochs)
    # floral
    flow_floral = FlowMatching(device=device, floral=True).to(device)
    train(flow_floral, train_loader, val_loader, device=device, epochs=epochs)


if __name__ == "__main__":
    # seed
    seed_everything()
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # loaders
    train_loader, val_loader = get_dataloaders(n_samples=1500, batch_size=64)
    # compare
    compare(train_loader=train_loader, val_loader=val_loader, epochs=100, device=device)
