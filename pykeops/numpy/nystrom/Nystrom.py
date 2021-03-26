import numpy as np
import pykeops
import numbers

from pykeops.numpy import LazyTensor as LazyTensor_n
from pykeops.numpy.cluster import grid_cluster
from pykeops.numpy.cluster import from_matrix
from pykeops.numpy.cluster import cluster_ranges_centroids, cluster_ranges
from pykeops.numpy.cluster import sort_clusters

from pykeops.torch import LazyTensor

# For LinearOperator math
from scipy.sparse.linalg import aslinearoperator, eigsh
from scipy.sparse.linalg.interface import IdentityOperator


class Nystrom_NK:
    """
    Class to implement Nystrom using numpy and PyKeops.
    * The fit method computes K^{-1}_q.
    * The transform method maps the data into the feature space underlying
    the Nystrom-approximated kernel.
    * The method K_approx directly computes the Nystrom approximation.
    Parameters:
    n_components [int] = how many samples to select from data.
    kernel [str] = type of kernel to use. Current options = {rbf:Gaussian,
                                                             exp: exponential}.
    sigma [float] = exponential constant for the RBF kernel.
    exp_sigma [float] = exponential constant for the exponential kernel.
    eps[float] = size for square bins in block-sparse preprocessing.
    k_means[int] = number of centroids for KMeans algorithm in block-sparse
                   preprocessing.
    n_iter[int] = number of iterations for KMeans
    dtype[type] = type of data: np.float32 or np.float64
    inv_eps[float] = additive invertibility constant for matrix decomposition.
    backend[string] = "GPU" or "CPU" mode
    verbose[boolean] = set True to print details
    random_state=[None, float] = to set a random seed for the random
                                 sampling of the samples. To be used when
                                 reproducibility is needed.
    """

    def __init__(
        self,
        n_components=100,
        kernel="rbf",
        sigma: float = 1.0,
        exp_sigma: float = 1.0,
        eps: float = 0.05,
        mask_radius: float = None,
        k_means=10,
        n_iter: int = 10,
        inv_eps: float = None,
        dtype=np.float32,
        backend=None,
        verbose=False,
        random_state=None,
    ):

        self.n_components = n_components
        self.kernel = kernel
        self.random_state = random_state
        self.sigma = sigma
        self.exp_sigma = exp_sigma
        self.eps = eps
        self.mask_radius = mask_radius
        self.k_means = k_means
        self.n_iter = n_iter
        self.dtype = dtype
        self.verbose = verbose

        if not backend:
            self.backend = "GPU" if pykeops.config.gpu_available else "CPU"
        else:
            self.backend = backend

        if inv_eps:
            self.inv_eps = inv_eps
        else:
            self.inv_eps = 1e-8

        if not mask_radius:
            if kernel == "rbf":
                self.mask_radius = 2 * np.sqrt(2) * self.sigma
            elif kernel == "exp":
                self.mask_radius = 8 * self.exp_sigma

    def fit(self, x: np.ndarray):
        """
        Args:   x = numpy array of shape (n_samples, n_features)
        Returns: Fitted instance of the class
        """
        if self.verbose:
            print(f"Working with backend = {self.backend}")

        # Basic checks
        assert type(x) == np.ndarray, "Input to fit(.) must be an array."
        assert (
            x.shape[0] >= self.n_components
        ), "The application needs X.shape[0] >= n_components."
        assert self.exp_sigma > 0, "Should be working with decaying exponential."

        # Update dtype
        self._update_dtype(x)
        # Number of samples
        n_samples = x.shape[0]
        # Define basis
        rnd = self._check_random_state(self.random_state)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[: self.n_components]
        basis = x[basis_inds]
        # Build smaller kernel
        basis_kernel = self._pairwise_kernels(basis, dense=False)
        # Spectral decomposition
        S, U = self._spectral(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), U.T)
        self.components_ = basis
        self.component_indices_ = inds

        return self

    def _spectral(self, X_i: LazyTensor):
        """
        Helper function to compute eigendecomposition of K_q.
        Written using LinearOperators which are lazy
        representations of sparse and/or structured data.
        Args:
            X_i[numpy LazyTensor]
        Returns
            S[np.array] eigenvalues,
            U[np.array] eigenvectors
        """
        K_linear = aslinearoperator(X_i)
        # K <- K + eps
        K_linear = (
            K_linear + IdentityOperator(K_linear.shape, dtype=self.dtype) * self.inv_eps
        )
        k = K_linear.shape[0] - 1
        S, U = eigsh(K_linear, k=k, which="LM")

        return S, U

    def transform(self, x: np.ndarray) -> np.array:
        """Applies transform on the data.

        Args:
            X [np.array] = data to transform
        Returns
            X [np.array] = data after transformation
        """

        K_nq = self._pairwise_kernels(x, self.components_, dense=True)
        x_new = K_nq @ self.normalization_.T
        return x_new

    def K_approx(self, x: np.array) -> np.array:
        """Function to return Nystrom approximation to the kernel.

        Args:
            X[np.array] = data used in fit(.) function.
        Returns
            K[np.array] = Nystrom approximation to kernel"""

        K_nq = self._pairwise_kernels(x, self.components_, dense=True)
        # For arrays: K_approx = K_nq @ K_q_inv @ K_nq.T
        # But to use @ with lazy tensors we have:
        K_q_inv = self.normalization_.T @ self.normalization_
        K_approx = K_nq @ (K_nq @ K_q_inv).T
        return K_approx.T

    def _pairwise_kernels(
        self, x: np.array, y: np.array = None, dense: bool = False
    ) -> LazyTensor:
        """Helper function to build kernel

        Args:   x[np.array] = data
                y[np.array] = array
                dense[bool] = False to work with lazy tensor reduction,
                              True to work with dense arrays
        Returns:
                K_ij[LazyTensor] if dense = False
                K_ij[np.array] if dense = True

        """
        if y is None:
            y = x
        if self.kernel == "rbf":
            x /= self.sigma
            y /= self.sigma
            if dense:
                x_i, x_j = x[:, None, :], y[None, :, :]
                K_ij = np.exp(-(((x_i - x_j) ** 2).sum(axis=2)))
            else:
                x_i, x_j = LazyTensor_n(x[:, None, :]), LazyTensor_n(y[None, :, :])
                K_ij = (-(((x_i - x_j) ** 2).sum(dim=2))).exp()
                # block-sparse reduction preprocess
                K_ij = self._Gauss_block_sparse_pre(x, y, K_ij)
        elif self.kernel == "exp":
            x /= self.exp_sigma
            y /= self.exp_sigma
            if dense:
                x_i, x_j = x[:, None, :], y[None, :, :]
                K_ij = np.exp(-np.sqrt((((x_i - x_j) ** 2).sum(axis=2))))
            else:
                x_i, x_j = LazyTensor_n(x[:, None, :]), LazyTensor_n(y[None, :, :])
                K_ij = (-(((x_i - x_j) ** 2).sum(-1)).sqrt()).exp()
                # block-sparse reduction preprocess
                K_ij = self._Gauss_block_sparse_pre(x, y, K_ij)  # TODO

        if not dense:
            K_ij.backend = self.backend

        return K_ij

    def _Gauss_block_sparse_pre(self, x: np.array, y: np.array, K_ij: LazyTensor):
        """
        Helper function to preprocess data for block-sparse reduction
        of the Gaussian kernel

        Args:
            x[np.array], y[np.array] = arrays giving rise to Gaussian kernel K(x,y)
            K_ij[LazyTensor_n] = symbolic representation of K(x,y)
            eps[float] = size for square bins
        Returns:
            K_ij[LazyTensor_n] = symbolic representation of K(x,y) with
                                set sparse ranges
        """
        # labels for low dimensions
        if x.shape[1] < 4 or y.shape[1] < 4:
            x_labels = grid_cluster(x, self.eps)
            y_labels = grid_cluster(y, self.eps)
            # range and centroid per class
            x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels)
            y_ranges, y_centroids, _ = cluster_ranges_centroids(y, y_labels)
        else:
            # labels for higher dimensions
            x_labels, x_centroids = self._KMeans(x)
            y_labels, y_centroids = self._KMeans(y)
            # compute ranges
            x_ranges = cluster_ranges(x_labels)
            y_ranges = cluster_ranges(y_labels)

        # sort points
        x, x_labels = sort_clusters(x, x_labels)
        y, y_labels = sort_clusters(y, y_labels)
        # Compute a coarse Boolean mask:
        if self.kernel == "rbf":
            D = np.sum((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2, 2)
        elif self.kernel == "exp":
            D = np.sqrt(
                np.sum((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2, 2)
            )
        keep = D < (self.mask_radius) ** 2
        # mask -> set of integer tensors
        ranges_ij = from_matrix(x_ranges, y_ranges, keep)
        K_ij.ranges = ranges_ij  # block-sparsity pattern

        return K_ij

    def _KMeans(self, x: np.array):
        """KMeans with Pykeops to do binning of original data.
        Args:
            x[np.array] = data
            k_means[int] = number of bins to build
            n_iter[int] = number iterations of KMeans loop
        Returns:
            labels[np.array] = class labels for each point in x
            clusters[np.array] = coordinates for each centroid
        """
        N, D = x.shape
        clusters = np.copy(x[: self.k_means, :])  # initialization of clusters
        x_i = LazyTensor_n(x[:, None, :])

        for i in range(self.n_iter):

            clusters_j = LazyTensor_n(clusters[None, :, :])
            D_ij = ((x_i - clusters_j) ** 2).sum(-1)  # points-clusters kernel
            labels = (
                D_ij.argmin(axis=1).astype(int).reshape(N)
            )  # Points -> Nearest cluster
            Ncl = np.bincount(labels).astype(self.dtype)  # Class weights
            for d in range(D):  # Compute the cluster centroids with np.bincount:
                clusters[:, d] = np.bincount(labels, weights=x[:, d]) / Ncl

        return labels, clusters

    def _update_dtype(self, x):
        """Helper function that sets dtype to that of
            the given data in the fitting step.

        Args:
            x [np.array] = raw data to remap
        Returns:
            nothing
        """
        self.dtype = x.dtype
        self.inv_eps = np.array([self.inv_eps]).astype(self.dtype)[0]

    def _check_random_state(self, seed):
        """Set/get np.random.RandomState instance for permutation

        Args
            seed[None, int]
        Returns:
            numpy random state
        """
        if seed is None:
            return np.random.mtrand._rand
        elif type(seed) == int:
            return np.random.RandomState(seed)
        raise ValueError(f"Seed {seed} must be None or an integer.")
