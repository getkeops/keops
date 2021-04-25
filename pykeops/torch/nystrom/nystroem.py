import torch

from pykeops.common.nystrom_generic import GenericNystroem


class Nystroem(GenericNystroem):
    """
    Nystroem class to work with Pytorch tensors.
    """

    def __init__(
        self,
        n_components=100,
        kernel="rbf",
        sigma: float = None,
        inv_eps: float = None,
        verbose=False,
        random_state=None,
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
        """

        super().__init__(n_components, kernel, sigma, inv_eps, verbose, random_state)
        from pykeops.torch.utils import torchtools
        from pykeops.torch import LazyTensor

        self.tools = torchtools
        self.verbose = verbose
        self.lazy_tensor = LazyTensor

    def _decomposition_and_norm(self, basis_kernel) -> torch.tensor:
        """
        Function to return self.normalization used in fit(.) function

        Args:
          basis_kernel: torch.tensor: K_q smaller sampled kernel

        Returns:
          K_q^{-1/2}: torch.tensor
        """

        U, S, V = torch.linalg.svd(
            basis_kernel, full_matrices=False
        )  # (Q,Q), (Q,), (Q,Q)
        S = torch.maximum(S, torch.ones(S.size()) * 1e-12)
        return torch.mm(U / torch.sqrt(S), V)  # (Q,Q)

    def _get_kernel(self, x: torch.tensor, y: torch.tensor, kernel=None):
        """
        Constructs dense kernel.

        Returns:
          K: torch.tensor: dense kernel

        """
        D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
        D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
        D_xy = D_xx - 2 * D_xy + D_yy  # (N,M)
        if kernel == "exp":
            D_xy = torch.sqrt(D_xy)
        return (-D_xy).exp()  # (N,M)

    def _update_dtype(self):
        "Overloading function to bypass in this subclass"
        pass

    def K_approx(self, X: torch.tensor) -> "K_approx operator":
        """Function to return Nystrom approximation to the kernel.

        Args:
          X: torch.tensor: data used in fit(.) function.
        Returns
          K_approx: K_approx_operator: Nystrom approximation to kernel
            which can be applied downstream as K_approx @ v for some 1d
            tensor v
        """

        K_nq = self._pairwise_kernels(X, self.components_, dense=False)  # (N, Q)
        K_approx = K_approx_operator(K_nq, self.normalization)  # (N, B), with v[N, B]
        return K_approx


class K_approx_operator:
    """Helper class to return K_approx as an object
    compatible with @ symbol
    """

    def __init__(self, K_nq, normalization):

        self.K_nq = K_nq  # dim: number of samples x num_components
        self.normalization = normalization

    def __matmul__(self, v: torch.tensor) -> torch.tensor:

        K_qn = self.K_nq.T
        self.K_nq.backend = "GPU_2D"

        x = K_qn @ v  # (Q,N), (N,B)
        x = self.normalization @ self.normalization.T @ x  # (Q,Q), (Q,Q), (Q, B)
        x = self.K_nq @ x  # (N,Q), (Q,B)
        return x  # (N,B)
