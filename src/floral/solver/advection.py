""" One-D advection solver
Author: Sahil Bhola, University of Michigan, 2025
"""

import torch
import numpy as np
from floral.utils import check_tensor_blowup
from tqdm import tqdm


class Advection:
    """Advection solver.
    Attributes:
    """

    def __init__(
        self,
        beta: float = 0.05,
        Nx: int = 64,
        Nt: int = 64,
        T: float = 1.0,
        Lx: float = 1.0,
        n_modes: int = 2,
        n_samples: int = 100,
        kmax: int = 8,
        use_ic: np.ndarray | None = None,
    ):
        self.beta = beta
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
        self.cfl = abs(self.beta) * self.dt / self.dx
        print(f"CFL: {self.cfl: .3e}")
        assert self.cfl < 1.0, "CFL must be less than 1.0 for numerical stability"
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
        """compute rhs"""
        assert isinstance(u, torch.Tensor) and u.ndim == 1
        # u_im1 = torch.roll(u, 1)
        # u_ip1 = torch.roll(u, -1)
        # dudx = (u_ip1 - u_im1) / (2.0 * self.dx)
        if self.beta > 0:
            dudx = (u - torch.roll(u, 1)) / self.dx
        else:
            dudx = (torch.roll(u, -1) - u) / self.dx
        return -self.beta * dudx

    def step_rk3(self, u):
        """Third order strong stability preserving Rungu-Kutta (SSPRK3) method
        Reference: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        """
        k1 = self._rhs(u)
        k2 = self._rhs(u + self.dt * k1)
        k3 = self._rhs(u + self.dt * (0.25 * k1 + 0.25 * k2))
        un = u + (self.dt / 6.0) * (k1 + k2 + 4.0 * k3)
        return un

    def _check_residual(self, u):
        """check the residual
        u: (torch.Tensor): solution of shape (Nt+1, Nx)
        """
        # dudt (forward differencwe + backward difference)
        dudt = torch.zeros_like(u)
        dudt[:-1, :] = (u[1:, :] - u[:-1, :]) / self.dt
        dudt[-1, :] = (u[-1, :] - u[-2, :]) / self.dt
        # dudx (upwinding)
        if self.beta > 0:
            dudx = (u - torch.roll(u, shifts=1, dims=1)) / self.dx
        else:
            dudx = (torch.roll(u, shifts=-1, dims=1) - u) / self.dx

        # residual
        residual = dudt + self.beta * dudx

        print(f"residual norm: {torch.linalg.norm(residual)}")

    def step_euler(self, u):
        """Forward euler"""
        return u + self.dt * self._rhs(u)

    def comp_exact_solution(self, u0, t):
        assert isinstance(u0, np.ndarray) and u0.ndim == 1
        # Shift by beta*t with periodic wrapping
        shift = (self.x - self.beta * t) % self.Lx
        return np.interp(shift, self.x, u0, period=self.Lx)

    def solve_single(self, u0, check_resisual: bool = False):
        """Returns (nt + 1, nx) solution"""
        u = torch.tensor(u0, dtype=torch.float32, device=self.device)
        sol_exact = np.zeros((self.Nt + 1, self.Nx))
        sol_exact[0] = u.clone().cpu().numpy()
        sol = torch.zeros(
            (self.Nt + 1, self.Nx), dtype=torch.float32, device=self.device
        )
        sol[0] = u.clone()
        energy = torch.zeros(self.Nt + 1, device=self.device)
        energy[0] = 0.5 * torch.sum(sol[0] * sol[0])
        for ii in range(1, self.Nt + 1):
            u = self.step_euler(u)
            sol[ii] = u.clone()
            energy[ii] = 0.5 * torch.sum(sol[ii] * sol[ii])
            sol_exact[ii] = self.comp_exact_solution(sol_exact[0], ii * self.dt)
        # check the residual
        if check_resisual:
            self._check_residual(sol)
        return sol.cpu().numpy(), energy.cpu().numpy(), sol_exact

    def solve(self):
        """all solve"""
        all_uT = []
        all_energy = []
        all_uT_exact = []
        pbar = tqdm(range(self.n_samples), desc="Advection")
        for ii in pbar:
            u0 = self.u0[ii]
            sol, energy, sol_exact = self.solve_single(u0, check_resisual=False)
            all_uT.append(sol)
            all_energy.append(energy)
            all_uT_exact.append(sol_exact)
        pbar.close()
        all_uT = np.array(all_uT)
        all_uT_exact = np.array(all_uT_exact)
        all_energy = np.array(all_energy)

        # print(f"Error norm: {np.linalg.norm(all_uT - all_uT_exact): .2f}")
        return np.array(all_uT), np.array(all_energy)
