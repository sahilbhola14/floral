""" Solver utilities
Author: Sahil Bhola, University of Michigan, 2025
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def sample_grf_rbf(
    coords: np.ndarray,
    n_samples: int,
    length_scale: float = 0.1,
    sigma: float = 1.0,
    nugget: float = 1e-10,
    rng: Optional[np.random.Generator] = None,
    kl_decomp: Optional[tuple[np.ndarray, np.ndarray]] = None,
    num_modes: Optional[int] = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Sample realizations of a Gaussian Random Field (GRF) using a
    Radial Basis Function (squared-exponential / RBF) kernel.

    The covariance between two points x_i and x_j is defined as:

        k(x_i, x_j) = sigma^2 * exp(-||x_i - x_j||^2 / (2 * length_scale^2))

    Samples are drawn via the truncated Karhunen-Loève (KL) decomposition of
    the covariance matrix.  The full decomposition K = V Λ Vᵀ is computed once
    and the leading ``num_modes`` eigenpairs (largest eigenvalues) are retained:

        f ≈ V_r Λ_r^½ z,  z ~ N(0, I),  z ∈ ℝ^{num_modes}

    Args:
        coords: Spatial coordinates of shape (N, d) where N is the number of
            points and d is the spatial dimension.
        n_samples: Number of GRF realizations to draw.
        length_scale: Length scale of the RBF kernel. Controls the spatial
            correlation length of the field.
        sigma: Marginal standard deviation (amplitude) of the field.
        nugget: Small diagonal regularization added to the covariance matrix
            to ensure positive definiteness.
        rng: NumPy random Generator instance. If None, the default global
            generator is used.
        kl_decomp: Pre-computed (truncated) KL decomposition as
            (eigenvalues, eigenvectors) with shapes (num_modes,) and
            (N, num_modes). If provided, the covariance matrix is not rebuilt
            and num_modes is inferred from the stored shapes.
        num_modes: Number of leading KL modes to retain. Defaults to N
            (full decomposition). Ignored when kl_decomp is provided.

    Returns:
        samples: Array of shape (n_samples, N) containing the GRF realizations.
        kl_decomp: Tuple (eigenvalues, eigenvectors) of shapes (num_modes,) and
            (N, num_modes). Can be cached and passed back on subsequent calls to
            skip the eigendecomposition.
    """
    if rng is None:
        rng = np.random.default_rng()

    assert n_samples > 0, "n_samples must be a positive integer"

    if kl_decomp is None:
        assert coords.ndim == 2, f"coords must be 2-D (N, d), got shape {coords.shape}"
        assert length_scale > 0, "length_scale must be positive"
        assert sigma > 0, "sigma must be positive"
        assert nugget >= 0, "nugget must be non-negative"

        N = coords.shape[0]
        if num_modes is None:
            num_modes = N
        assert 0 < num_modes <= N, f"num_modes must be in (0, {N}], got {num_modes}"

        # pairwise squared distances: (N, N)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, d)
        sq_dists = np.sum(diff**2, axis=-1)  # (N, N)

        # RBF covariance matrix
        K = sigma**2 * np.exp(-sq_dists / (2.0 * length_scale**2))
        K += nugget * np.eye(N)

        # Full KL decomposition: eigenvalues sorted ascending by eigh
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        # clip small negatives from numerical noise
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        # retain the leading num_modes (largest eigenvalues are at the tail)
        eigenvalues = eigenvalues[-num_modes:]  # (num_modes,)
        eigenvectors = eigenvectors[:, -num_modes:]  # (N, num_modes)
        kl_decomp = (eigenvalues, eigenvectors)
    else:
        eigenvalues, eigenvectors = kl_decomp

    # N from eigenvectors rows; num_modes from eigenvalues length
    N = eigenvectors.shape[0]
    num_modes = eigenvalues.shape[0]

    # draw samples: f = V_r Λ_r^½ z, z ~ N(0, I), shape (n_samples, N)
    z = rng.standard_normal((n_samples, num_modes))
    samples = z @ (eigenvectors * np.sqrt(eigenvalues)).T

    assert samples.shape == (
        n_samples,
        N,
    ), f"Expected samples of shape ({n_samples}, {N}), got {samples.shape}"
    return samples, kl_decomp


def sample_periodic_noise(
    x: np.ndarray,
    Lx: float,
    n_samples: int,
    k_max: int = 8,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample periodic noise as a sum of random Fourier modes.

    noise(x) = sum_{k=1}^{k_max} (alpha_k * cos(2pi k x / Lx)
                                 + beta_k  * sin(2pi k x / Lx))

    alpha_k, beta_k ~ N(0, noise_std^2).  By construction every realization is
    periodic on [0, Lx].

    Args:
        x: Spatial grid of shape (N,).
        Lx: Domain length.
        n_samples: Number of noise realizations.
        k_max: Highest wavenumber included.
        noise_std: Standard deviation of the Fourier coefficients.
        rng: NumPy random Generator. If None, the default generator is used.

    Returns:
        noise: Array of shape (n_samples, N).
    """
    if rng is None:
        rng = np.random.default_rng()

    k = np.arange(1, k_max + 1)  # (k_max,)
    freq = 2.0 * np.pi * k / Lx  # (k_max,)

    alpha = rng.normal(0.0, noise_std, size=(n_samples, k_max))  # (B, k_max)
    beta = rng.normal(0.0, noise_std, size=(n_samples, k_max))  # (B, k_max)

    # (B, k_max) @ (k_max, N) -> (B, N)
    noise = alpha @ np.cos(freq[:, None] * x[None, :]) + beta @ np.sin(
        freq[:, None] * x[None, :]
    )
    return noise
