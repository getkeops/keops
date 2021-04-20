import torch

from pykeops.common.nystrom_generic import GenericNystrom
from pykeops.torch.utils import torchtools
from pykeops.torch import LazyTensor


class Nystrom(GenericNystrom):
    def __init__(
        self,
        n_components=100,
        kernel="rbf",
        sigma: float = None,
        eps: float = 0.05,
        mask_radius: float = None,
        k_means=10,
        n_iter: int = 10,
        inv_eps: float = None,
        verbose=False,
        random_state=None,
        tools=None,
    ):
        super().__init__(
            n_components,
            kernel,
            sigma,
            eps,
            mask_radius,
            k_means,
            n_iter,
            inv_eps,
            verbose,
            random_state,
        )

        self.tools = torchtools
        self.verbose = verbose
        self.LazyTensor = LazyTensor

    def _update_dtype(self, x):
        pass

    def _to_device(self, x):
        return x.to(self.device)

    def _decomposition_and_norm(self, basis_kernel):
        """Function to return self.nomalization_ used in fit(.) function
        Args:
            basis_kernel[torch LazyTensor] = subset of input data
        Returns:
            self.normalization_[torch.tensor]  X_q is the q x D-dimensional sub matrix of matrix X
        """
        basis_kernel = basis_kernel.to(
            self.device
        )  # dim: num_components x num_components
        U, S, V = torch.linalg.svd(basis_kernel, full_matrices=False)
        S = torch.maximum(S, torch.ones(S.size()).to(self.device) * 1e-12)
        return torch.mm(U / torch.sqrt(S), V)  # dim: num_components x num_components

    def K_approx(self, X: torch.tensor) -> "K_approx operator":
        """Function to return Nystrom approximation to the kernel.
        Args:
            X = data used in fit(.) function.
        Returns
            K_approx = Nystrom approximation to kernel which can be applied
                        downstream as K_approx @ v for some 1d tensor v"""

        K_nq = self._pairwise_kernels(X, self.components_, dense=False)
        K_approx = K_approx_operator(K_nq, self.normalization_)
        return K_approx

    def _KMeans(self, x: torch.tensor):
        """KMeans with Pykeops to do binning of original data.
        Args:
            x = data
        Returns:
            labels[np.array] = class labels for each point in x
            clusters[np.array] = coordinates for each centroid
        """

        N, D = x.shape
        clusters = torch.clone(x[: self.k_means, :])  # initialization of clusters
        x_i = LazyTensor(x[:, None, :])

        for i in range(self.n_iter):

            clusters_j = LazyTensor(clusters[None, :, :])
            D_ij = ((x_i - clusters_j) ** 2).sum(-1)  # points-clusters kernel
            labels = D_ij.argmin(axis=1).reshape(N)  # Points -> Nearest cluster
            Ncl = torch.bincount(labels)  # Class weights
            for d in range(D):  # Compute the cluster centroids with np.bincount:
                clusters[:, d] = torch.bincount(labels, weights=x[:, d]) / Ncl

        return labels, clusters


class K_approx_operator:
    """Helper class to return K_approx as an object
    compatible with @ symbol"""

    def __init__(self, K_nq, normalization):

        self.K_nq = K_nq  # dim: number of samples x num_components
        self.K_nq.backend = "GPU_2D"
        self.normalization = normalization

    def __matmul__(self, x: torch.tensor) -> torch.tensor:

        x = self.K_nq.T @ x
        x = self.normalization @ self.normalization.T @ x
        x = self.K_nq @ x
        return x
