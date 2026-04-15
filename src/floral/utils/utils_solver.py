""" Solver utilities
Author: Sahil Bhola, University of Michigan, 2025
"""

import numpy as np


def sample_grf_rbf(
    coords: np.ndarray,
    n_samples: int,
    length_scale: float = 0.1,
    sigma: float = 1.0,
    nugget: float = 1e-10,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Sample realizations of a Gaussian Random Field (GRF) using a
    Radial Basis Function (squared-exponential / RBF) kernel.

    The covariance between two points x_i and x_j is defined as:

        k(x_i, x_j) = sigma^2 * exp(-||x_i - x_j||^2 / (2 * length_scale^2))

    Samples are drawn via the Cholesky decomposition of the covariance matrix.

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

    Returns:
        samples: Array of shape (n_samples, N) containing the GRF realizations.
    """
    if rng is None:
        rng = np.random.default_rng()

    assert coords.ndim == 2, f"coords must be 2-D (N, d), got shape {coords.shape}"
    assert length_scale > 0, "length_scale must be positive"
    assert sigma > 0, "sigma must be positive"
    assert nugget >= 0, "nugget must be non-negative"
    assert n_samples > 0, "n_samples must be a positive integer"

    N = coords.shape[0]

    # pairwise squared distances: (N, N)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, d)
    sq_dists = np.sum(diff**2, axis=-1)  # (N, N)

    # RBF covariance matrix
    K = sigma**2 * np.exp(-sq_dists / (2.0 * length_scale**2))
    K += nugget * np.eye(N)

    # Cholesky factorization: K = L L^T
    L = np.linalg.cholesky(K)

    # draw samples: (n_samples, N)
    z = rng.standard_normal((n_samples, N))
    samples = z @ L.T

    assert samples.shape == (
        n_samples,
        N,
    ), f"Expected samples of shape ({n_samples}, {N}), got {samples.shape}"
    return samples
