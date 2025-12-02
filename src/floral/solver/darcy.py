""" Darcy solver
Author: Sahil Bhola, University of Michigan, 2025
Modified from: https://github.com/christian-jacobsen/CoCoGen/tree/master
for multi-fidelity training.
"""

import math
import numpy as np
import scipy
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from floral.utils import timeit, check_tensor_blowup


class Darcy:
    def __init__(
        self, resolution: int, ntheta: int = 128, threads: int = 32, n_samples: int = 10
    ):
        self.resolution = resolution
        self.ntheta = ntheta  # mode truncations
        self.threads = threads  # number of threads for parallelization
        self.n_samples = n_samples
        # mesh
        self.xv = np.linspace(0, 1, self.resolution)
        XX, YY = np.meshgrid(self.xv, self.xv, indexing="ij")
        self.mesh = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)
        self.domain = self.mesh.reshape(self.resolution, self.resolution, 2).transpose(
            2, 0, 1
        )
        # correlation matrix params
        self.corr_params = {"c0": 0.1, "sigma": 1.0, "thresh": 1e-12}  # original
        # source
        # self.source_strength = 10.0 # original
        self.source_strength = 50.0  # modified
        # A_matrix for darcy flow (with integral enforcement)
        self.A_reg = self._get_A_matrix()
        # permeability
        self.permeability_data = self._sample_permeability(n_samples=self.n_samples)

    def _set_permeability_data(self, permeability_data):
        for k in ["permeability", "generative_params", "eigenvalues"]:
            assert (
                permeability_data[k].shape == self.permeability_data[k].shape
            ), f"{k} has inconsistent shape"

        for k, v in permeability_data.items():
            self.permeability_data[k] = v

    def _get_A_matrix(self):
        """get the A matrix for the darcy flow with integral enforcement"""
        x = np.linspace(0, 1, self.resolution)
        dx = x[1] - x[0]
        nx = len(x)
        A = np.zeros((1, nx**2))
        j = 0
        i = 0
        for k in range(nx**2):
            if np.mod(k, nx) == 0:
                j += 1
                i = 1

            if (i == 1) and (j == 1):  # bottom left corner
                A[0, k] = 1
            elif (i == nx) and (j == nx):  # top right corner
                A[0, k] = 1
            elif (i == 1) and (j == nx):  # top left corner
                A[0, k] = 1
            elif (i == nx) and (j == 1):  # bottom right corner
                A[0, k] = 1
            elif i == 1:  # left boundary
                A[0, k] = 2
            elif j == 1:  # bottom boundary
                A[0, k] = 2
            elif i == nx:  # right boundary
                A[0, k] = 2
            elif j == nx:  # top boundary
                A[0, k] = 2
            else:  # interior
                A[0, k] = 4

        return A * dx**2 / 4

    def _get_correlation_matrix(self):
        """get the correlation matrix"""
        n = self.mesh.shape[0]
        c0 = self.corr_params["c0"] * np.ones((self.mesh.shape[1], 1))
        c0 = 1 / (c0**2)
        sigma = self.corr_params["sigma"]
        C = np.zeros((n, n))
        for i in range(n):
            # gaussian correlation
            point = self.mesh[i : i + 1, :]
            X = (self.mesh - point * np.ones((n, 2))) ** 2
            C[:, i] = sigma * np.exp(-np.sqrt(np.matmul(X, c0)))[:, 0]

        thresh = self.corr_params["thresh"]
        C[C < thresh] = 0

        return C

    def _get_random_field_lu(self):
        """compute the eigenvalues and eigenvectors of the covariance matrix"""
        # covariance matrix
        C = self._get_correlation_matrix()
        # compute the eigenvalues and eigenvectors
        S, U = scipy.sparse.linalg.eigs(C, k=self.ntheta)
        S_t, U_t = scipy.sparse.linalg.eigs(C, k=self.ntheta + 10)
        ev = np.abs(np.real(S))
        inds = np.flip(np.argsort(ev))
        ev = ev[inds]
        U = np.real(U)[:, inds]
        assert (
            ev.ndim == 1 and ev.shape[0] == self.ntheta
        ), "Eigen values dimension mismatch"
        assert (
            U.ndim == 2
            and U.shape[0] == self.resolution**2
            and U.shape[1] == self.ntheta
        ), "Eigen vectors dimension mismatch"
        return ev, U

    def _get_generative_params(self, n_samples):
        """get the generative parameters"""
        theta = np.random.rand(self.ntheta, n_samples) * 2.5
        # theta = np.ones((self.ntheta, n_samples)) * 2.5 # for testing purposes
        return theta

    def _sample_permeability(self, n_samples):
        """sample the permeabiilty field"""
        # get the eigen values and eigen vector os the covariance matrix
        ev, U = self._get_random_field_lu()
        # get the generative parameters
        theta = self._get_generative_params(n_samples)
        # construct permeability field (B, resolution, resolution)
        K = self._construct_permeability_from_UL(ev, U, theta)

        permeability_data = {
            "permeability": K,
            "generative_params": theta,
            "eigenvalues": ev,
            "eigenvectors": U,
        }
        return permeability_data

    def _construct_permeability_from_UL(
        self, ev: np.ndarray, U: np.ndarray, theta: np.ndarray
    ):
        """construct permeability field from eigen values,
        eigen vectors and generative parameters"""
        resolution = int(math.sqrt(U.shape[0]))
        assert resolution**2 == U.shape[0], "Resolution mismatch"
        assert (
            theta.ndim == 2 and theta.shape[0] == self.ntheta
        ), "Generative parameters dimension mismatch"
        assert (
            ev.ndim == 1 and ev.shape[0] == self.ntheta
        ), "Eigen values dimension mismatch"
        assert (
            U.ndim == 2 and U.shape[0] == resolution**2 and U.shape[1] == self.ntheta
        ), "Eigen vectors dimension mismatch"
        UL = U * np.sqrt(ev[np.newaxis, :])
        n_samples = theta.shape[1]
        K = np.exp(UL @ theta).T.reshape(n_samples, resolution, resolution)
        return K

    def grad_K(self, K, i, j, dx, dim):
        """compute gradient of K at location (i,j) in dimension dim"""
        dK = 0
        if dim == 1:
            dK = (K[i + 1, j] - K[i - 1, j]) / (2 * dx)
        elif dim == 2:
            dK = (K[i, j + 1] - K[i, j - 1]) / (2 * dx)
        return dK

    def form_matrix(self, K, xv, dx):
        """Form the matrix A and the source vector f for the Darcy flow problem."""
        nx = len(xv)
        A = np.zeros((nx**2, nx**2))
        f = np.zeros((nx**2, 1))

        j = -1
        i = 0

        dx2 = dx**2
        for k in range(nx**2):
            if (np.mod(k, nx)) == 0:
                j += 1
                i = 0

            A[k, k] = K[i, j] * 4 / (dx2)

            # corners
            if (i == 0) and (j == 0):
                A[k, k + 1] = -2 * K[i, j] / (dx2)
                A[k, k + nx] = -2 * K[i, j] / (dx2)
            elif (i == (nx - 1)) and (j == (nx - 1)):
                A[k, k - 1] = -2 * K[i, j] / (dx2)
                A[k, k - nx] = -2 * K[i, j] / (dx2)
            elif (i == 0) and (j == (nx - 1)):
                A[k, k + 1] = -2 * K[i, j] / (dx2)
                A[k, k - nx] = -2 * K[i, j] / (dx2)
            elif (i == (nx - 1)) and (j == 0):
                A[k, k - 1] = -2 * K[i, j] / (dx2)
                A[k, k + nx] = -2 * K[i, j] / (dx2)

            # bondaries
            elif (i == 0) or (j == 0) or (i == (nx - 1)) or (j == (nx - 1)):
                if i == 0:
                    gK = self.grad_K(K, i, j, dx, 2) / (2 * dx)
                    A[k, k + 1] = -2 * K[i, j] / dx2
                    A[k, k + nx] = -K[i, j] / dx2 - gK
                    A[k, k - nx] = -K[i, j] / dx2 + gK
                elif i == (nx - 1):
                    gK = self.grad_K(K, i, j, dx, 2) / (2 * dx)
                    A[k, k - 1] = -2 * K[i, j] / dx2
                    A[k, k + nx] = -K[i, j] / dx2 - gK
                    A[k, k - nx] = -K[i, j] / dx2 + gK
                elif j == 0:
                    gK = self.grad_K(K, i, j, dx, 1) / (2 * dx)
                    A[k, k + nx] = -2 * K[i, j] / dx2
                    A[k, k - 1] = -K[i, j] / dx2 + gK
                    A[k, k + 1] = -K[i, j] / dx2 - gK
                elif j == (nx - 1):
                    gK = self.grad_K(K, i, j, dx, 1) / (2 * dx)
                    A[k, k - nx] = -2 * K[i, j] / dx2
                    A[k, k - 1] = -K[i, j] / dx2 + gK
                    A[k, k + 1] = -K[i, j] / dx2 - gK

            # interior
            else:
                gK1 = self.grad_K(K, i, j, dx, 1) / (2 * dx)
                gK2 = self.grad_K(K, i, j, dx, 2) / (2 * dx)
                fac = -K[i, j] / dx2
                A[k, k - 1] = fac + gK1
                A[k, k + 1] = fac - gK1
                A[k, k + nx] = fac - gK2
                A[k, k - nx] = fac + gK2

            x, y = xv[i], xv[j]

            # source function
            if (np.abs(x - 0.0625) <= 0.0625) and (np.abs(y - 0.0625) <= 0.0625):
                f[k, 0] = self.source_strength
            elif (np.abs(x - 1 + 0.0625) <= 0.0625) and (
                np.abs(y - 1 + 0.0625) <= 0.0625
            ):
                f[k, 0] = -self.source_strength

            i += 1
        return A, f

    def compute_u(self, P, K, nx, xv, dx):
        U1, U2 = np.zeros(P.shape), np.zeros(P.shape)
        for i in range(nx):
            for j in range(nx):
                if (
                    ((j == 0) or (j == (nx - 1))) and (i != 0) and (i != (nx - 1))
                ):  # bottom or top boundary (no corners)
                    U1[i, j] = -K[i, j] * (P[i + 1, j] - P[i - 1, j]) / (2 * dx)
                elif (
                    ((i == 0) or (i == (nx - 1))) and (j != 0) and (j != (nx - 1))
                ):  # left or right boundary
                    U2[i, j] = -K[i, j] * (P[i, j + 1] - P[i, j - 1]) / (2 * dx)
                elif (
                    (i == 0 and j == 0)
                    or (i == (nx - 1) and j == (nx - 1))
                    or (i == 0 and j == (nx - 1))
                    or (j == 0 and i == (nx - 1))
                ):  # corners
                    U1[i, j] = 0
                    U2[i, j] = 0
                else:  # interior
                    U1[i, j] = -K[i, j] * (P[i + 1, j] - P[i - 1, j]) / (2 * dx)
                    U2[i, j] = -K[i, j] * (P[i, j + 1] - P[i, j - 1]) / (2 * dx)

        return U1, U2

    def _solve(self, K: np.ndarray, sample_index: int, sparse_solve: bool = True):
        if sample_index % 1000 == 0:
            print(f"Generating sample {sample_index}...")
        xv = np.linspace(0, 1, self.resolution)
        dx = xv[1] - xv[0]
        A, f = self.form_matrix(K, xv, dx)
        if sparse_solve:
            # Form KKT matrix
            n = A.shape[0]
            KKT = np.zeros((n + 1, n + 1))
            KKT[:n, :n] = A
            KKT[:n, n] = self.A_reg.flatten()  # or self.A_reg.ravel()
            KKT[n, :n] = self.A_reg.T.flatten()

            # Form KKT right-hand side
            rhs = np.zeros((n + 1, 1))
            rhs[:n] = f
            KKT_sparse = csr_matrix(KKT)
            solution = spsolve(KKT_sparse, rhs.flatten())
            P = solution[:n]
            P = P.reshape((self.resolution, self.resolution))
        else:
            A = np.concatenate((A, self.A_reg), axis=0)
            f = np.concatenate((f, np.zeros((1, 1))), axis=0)
            P, _, _, _ = scipy.linalg.lstsq(A, f)
            P = P.reshape((self.resolution, self.resolution))
        U1, U2 = self.compute_u(P, K, self.resolution, xv, dx)

        return P, U1, U2

    @timeit(func_name="Darcy solve")
    def solve(self):
        """solve the darcy flow problem"""

        K = self.permeability_data["permeability"]

        results = Parallel(n_jobs=self.threads)(
            delayed(self._solve)(K[i], i) for i in range(0, self.n_samples)
        )

        P, U1, U2 = zip(*results)
        P = np.stack(P)  # high-fidelity pressure field
        U1 = np.stack(U1)  # high-fidelity velocity field in x-direction
        U2 = np.stack(U2)  # high-fidelity velocity field in y-direction
        # check blowup
        check_tensor_blowup(P, "pressure")
        check_tensor_blowup(U1, "x-velocity")
        check_tensor_blowup(U2, "y-velocity")
        # data dict
        data_dict = {
            "pressure": P,
            "permeability": K,
            "velocity_x": U1,
            "velocity_y": U2,
        }
        # check shape
        for k, v in data_dict.items():
            assert v.shape == (
                self.n_samples,
                self.resolution,
                self.resolution,
            ), f"{k} has incorrect shape"
        return data_dict
