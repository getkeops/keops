import numpy as np

from scipy.linalg import eigh
from scipy.sparse.linalg import aslinearoperator

from pykeops.common.nystrom_generic import GenericNystroem

from typing import List


class Nystroem(GenericNystroem):
    """
    Nystroem class to work with Numpy arrays.
    """

    def __init__(
        self,
        n_components=100,
        kernel="rbf",
        sigma: float = None,
        inv_eps: float = None,
        verbose=False,
        random_state=None,
        eigvals: List[int] = None,
    ):

        """
        Args:
             n_components: int: how many samples to select from data.
            kernel: str: type of kernel to use. Current options = {rbf:Gaussian,
                exp: exponential}.
            sigma: float: exponential constant for the RBF and exponential kernels.
            inv_eps: float: additive invertibility constant for matrix decomposition.
            verbose: boolean: set True to print details.
            random_state: int: to set a random seed for the random sampling of the
                samples. To be used when reproducibility is needed.
            eigvals: eigenvalues index interval [a,b] for constructed K_q,
                where 0 <= a < b < length of K_q

        """
        super().__init__(n_components, kernel, sigma, inv_eps, verbose, random_state)
        from pykeops.numpy.utils import numpytools
        from pykeops.numpy import LazyTensor

        self.tools = numpytools
        self.lazy_tensor = LazyTensor
        self.eigvals = eigvals

        if eigvals:
            assert eigvals[0] < eigvals[1], "eigvals = [a,b] needs a < b"
            assert (
                eigvals[1] < n_components
            ), "max eigenvalue index needs to be less\
            than size of K_q = n_components"

    def _decomposition_and_norm(self, X: np.array) -> np.array:
        """
        Computes K_q^{-1/2}.

        Returns:
            K_q^{-1/2}: np.array
        """

        X = (
            X + np.eye(X.shape[0], dtype=self.dtype) * self.inv_eps
        )  # (Q,Q)  Q - num_components
        S, U = eigh(X, eigvals=self.eigvals)  # (Q,), (Q,Q)
        S = np.maximum(S, 1e-12)

        return np.dot(U / np.sqrt(S), U.T)  # (Q,Q)

    def _get_kernel(self, x: np.array, y: np.array, kernel=None) -> np.array:

        D_xx = np.sum((x ** 2), axis=-1)[:, None]  # (N,1)
        D_xy = x @ y.T  # (N,D) @ (D,M) = (N,M)
        D_yy = np.sum((y ** 2), axis=-1)[None, :]  # (1,M)
        D_xy = D_xx - 2 * D_xy + D_yy  # (N,M)
        if kernel == "exp":
            D_xy = np.sqrt(D_xy)
        return np.exp(-D_xy)  # (N,M)

    def K_approx(self, x: np.array) -> "LinearOperator":
        """
        Method to return Nystrom approximation to the kernel.

        Args:
            x: np.array: data used in fit(.) function.
        Returns
            K: LinearOperator: Nystrom approximation to kernel
        """
        K_nq = self._pairwise_kernels(x, self.components_, dense=False)  # (N, Q)

        K_qn = K_nq.T
        K_nq.backend = "GPU_2D"
        K_qn = aslinearoperator(K_qn)
        K_nq = aslinearoperator(K_nq)

        K_q_inv = self.normalization.T @ self.normalization  # (Q,Q)
        K_q_inv = aslinearoperator(K_q_inv)

        K_approx = K_nq @ K_q_inv @ K_qn  # (N,Q), (Q,Q), (Q,N)

        return K_approx  # (N, N)
