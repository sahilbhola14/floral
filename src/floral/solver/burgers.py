""" One-D Burgers solver
Author: Sahil Bhola, University of Michigan, 2025
"""

import torch
import numpy as np
from floral.utils import check_tensor_blowup
from tqdm import tqdm


class Burgers:
    """Viscous Burgers' solver.
    Attributes:
    """

    def __init__(
        self,
        nu: float = 0.01,
        Nx: int = 64,
        Nt: int = 64,
        T: float = 0.2,
        Lx: float = 1.0,
        n_modes: int = 2,
        n_samples: int = 100,
        kmax: int = 8,
        use_ic: np.ndarray | None = None,
    ):
        self.nu = nu
        self.Nx = Nx
        self.Nt = Nt
        self.T = T
        self.Lx = Lx
        self.n_modes = n_modes
        self.n_samples = n_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"running solve on: {self.device}")

        self.dt = self.T / self.Nt
        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        self.t = np.arange(0, self.Nt + 1) * self.dt
        self.dx = self.x[1] - self.x[0]
        XX, TT = np.meshgrid(self.x, self.t[1:])
        self.domain = np.concatenate((XX.reshape(-1, 1), TT.reshape(-1, 1)), axis=1)
        print(f"dt: {self.dt: .4f} dx: {self.dx: .4f}")

        # initial conditions
        if use_ic is None:
            print("Computing IC")
            self.u0 = self.get_initial_conditions(
                kmax=kmax, n_modes=n_modes, n_samples=self.n_samples
            )
            assert isinstance(self.u0, np.ndarray) and self.u0.shape == (
                self.n_samples,
                self.Nx,
            )
        else:
            print("Using specified IC")
            assert isinstance(use_ic, np.ndarray) and use_ic.shape == (
                self.n_samples,
                self.Nx,
            )
            self.u0 = use_ic
        # check blow up
        check_tensor_blowup(self.u0, "Initial condition")

    def get_initial_conditions(self, kmax: int, n_modes: int, n_samples: int):
        """initial condition"""
        np.random.seed(42)

        n_j = np.random.randint(1, kmax + 1, size=(n_samples, n_modes))
        wave_number = 2 * np.pi * n_j / self.Lx

        amp = np.random.uniform(0, 1, size=(n_samples, n_modes))
        phase = np.random.uniform(0, 2 * np.pi, size=(n_samples, n_modes))

        u0 = np.sum(
            amp[:, :, None]
            * np.sin(wave_number[:, :, None] * self.x + phase[:, :, None]),
            axis=1,
        )

        # random abs
        mask = np.random.rand(n_samples) < 0.1
        u0[mask] = np.abs(u0[mask])
        # random sign flip
        mask = np.random.rand(n_samples) < 0.5
        u0[mask] = -u0[mask]
        # windowing
        mask = np.random.rand(n_samples) < 0.1
        n_w = mask.sum()
        if n_w > 0:
            xL = np.random.uniform(0.1, 0.45, size=n_w)
            xR = np.random.uniform(0.55, 0.9, size=n_w)
            trns = 0.1
            window = 0.5 * (
                np.tanh((self.x - xL[:, None]) / trns)
                - np.tanh((self.x - xR[:, None]) / trns)
            )
            u0[mask] *= window

        return u0

    def set_initial_condition(self, use_ic):
        assert isinstance(use_ic, np.ndarray) and use_ic.shape == (
            self.n_samples,
            self.Nx,
        )
        self.u0 = use_ic

    def _rhs(self, u):
        """compute rhs
        For advection term, second order upwind is used and for diffusion terms we use
        central difference.
        """
        assert isinstance(u, torch.Tensor) and u.ndim == 1
        u_ip1 = torch.roll(u, -1)
        u_ip2 = torch.roll(u, -2)
        u_im1 = torch.roll(u, 1)
        u_im2 = torch.roll(u, 2)

        dx = self.dx

        # Upwind based on sign(u)
        pos_mask = (u > 0).float()
        neg_mask = 1.0 - pos_mask

        dudx_pos = (3 * u - 4 * u_im1 + u_im2) / (2 * dx)
        dudx_neg = (-3 * u + 4 * u_ip1 - u_ip2) / (2 * dx)

        dudx = pos_mask * dudx_pos + neg_mask * dudx_neg

        # diffusion term
        d2udx2 = (u_ip1 - 2 * u + u_im1) / (dx * dx)

        dudt = self.nu * d2udx2 - u * dudx
        return dudt

    def step_rk3(self, u):
        """Third order strong stability preserving Rungu-Kutta (SSPRK3) method
        Reference: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        """
        k1 = self._rhs(u)
        k2 = self._rhs(u + self.dt * k1)
        k3 = self._rhs(u + self.dt * (0.25 * k1 + 0.25 * k2))
        un = u + (self.dt / 6.0) * (k1 + k2 + 4.0 * k3)
        return un

    def solve_single(self, u0):
        """Returns (nt + 1, nx) solution"""
        u = torch.tensor(u0, dtype=torch.float32, device=self.device)
        sol = torch.zeros(
            (self.Nt + 1, self.Nx), dtype=torch.float32, device=self.device
        )
        sol[0] = u.clone()
        energy = torch.zeros(self.Nt + 1, device=self.device)
        energy[0] = 0.5 * torch.sum(sol[0] * sol[0])
        for ii in range(1, self.Nt + 1):
            u = self.step_rk3(u)
            sol[ii] = u.clone()
            energy[ii] = 0.5 * torch.sum(sol[ii] * sol[ii])
        return sol.cpu().numpy(), energy.cpu().numpy()

    def solve(self):
        """all solve"""
        all_uT = []
        all_energy = []
        pbar = tqdm(range(self.n_samples), desc="Burgers")
        for ii in pbar:
            u0 = self.u0[ii]
            sol, energy = self.solve_single(u0)
            all_uT.append(sol)
            all_energy.append(energy)
        pbar.close()
        return np.array(all_uT), np.array(all_energy)
