import numpy as np
import pykeops
from typing import TypeVar, Union, Tuple
import warnings

# Generic placeholder for numpy and torch variables.
generic_array = TypeVar("generic_array")
GenericLazyTensor = TypeVar("GenericLazyTensor")


class GenericNystrom:
    """Super class defining the Nystrom operations. The end user should
    use numpy.nystrom or torch.nystrom subclasses."""

    def __init__(
        self,
        n_components: int = 100,
        kernel: Union[str, callable] = "rbf",
        sigma: float = None,
        eps: float = 0.05,
        mask_radius: float = None,
        k_means: int = 10,
        n_iter: int = 10,
        inv_eps: float = None,
        verbose: bool = False,
        random_state: Union[None, int] = None,
        tools=None,
    ):

        """
        n_components  = how many samples to select from data.
        kernel  = type of kernel to use. Current options = {rbf:Gaussian,
                                                                 exp: exponential}.
        sigma  = exponential constant for the RBF and exponential kernels.
        eps = size for square bins in block-sparse preprocessing.
        k_means = number of centroids for KMeans algorithm in block-sparse
                       preprocessing.
        n_iter = number of iterations for KMeans.
        dtype = type of data: np.float32 or np.float64
        inv_eps = additive invertibility constant for matrix decomposition.
        verbose = set True to print details.
        random_state = to set a random seed for the random sampling of the samples.
                        To be used when  reproducibility is needed.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma
        self.eps = eps
        self.mask_radius = mask_radius
        self.k_means = k_means
        self.n_iter = n_iter
        self.dtype = None
        self.verbose = verbose
        self.random_state = random_state
        self.tools = None
        self.LazyTensor = None

        self.device = "cuda" if pykeops.config.gpu_available else "cpu"

        if inv_eps:
            self.inv_eps = inv_eps
        else:
            self.inv_eps = 1e-8

    def fit(self, x: generic_array) -> "GenericNystrom":
        """
        Args:   x = array or tensor of shape (n_samples, n_features)
        Returns: Fitted instance of the class
        """
        x = self._to_device(x)
        self.dtype = x.dtype

        # Basic checks
        assert self.tools.is_tensor(
            x
        ), "Input to fit(.) must be an array\
        if using numpy and tensor if using torch."
        assert (
            x.shape[0] >= self.n_components
        ), "The application needs\
        X.shape[0] >= n_components."
        if self.kernel == "exp" and not (self.sigma is None):
            assert self.sigma > 0, "Should be working with decaying exponential."

        # Set default sigma
        # if self.sigma is None and self.kernel == 'rbf':
        if self.sigma is None:
            self.sigma = np.sqrt(x.shape[1])

        if self.mask_radius is None:
            if self.kernel == "rbf":
                # TODO get mask_radius correct
                self.mask_radius = 8 * self.sigma
            elif self.kernel == "exp":
                self.mask_radius = 8 * self.sigma

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
        basis_kernel = self._pairwise_kernels(basis, dense=True)
        # Decomposition is an abstract method that needs to be defined in each class
        self.normalization_ = self._decomposition_and_norm(basis_kernel)
        self.components_ = basis
        self.component_indices_ = inds

        return self

    def _decomposition_and_norm(self, X: GenericLazyTensor):
        """
        To be defined in the subclass
        """
        raise NotImplementedError(
            "Subclass must implement the method _decomposition_and_norm."
        )

    def transform(self, x: generic_array, dense=True) -> generic_array:
        """
        Applies transform on the data mapping it to the feature space
        which supports the approximated kernel.
        Args:
            X = data to transform
        Returns
            X = data after transformation
        """
        if type(x) == np.ndarray and not dense:
            warnings.warn("For Numpy transform it is best to use dense=True")

        x = self._to_device(x)
        K_nq = self._pairwise_kernels(x, self.components_, dense=dense)
        x_new = K_nq @ self.normalization_
        return x_new

    def _pairwise_kernels(self, x: generic_array, y: generic_array = None, dense=False):
        """Helper function to build kernel
        Args:   x[np.array or torch.tensor] = data
                y[np.array or torch.tensor] = array/tensor
                dense[bool] = False to work with lazy tensor reduction,
                              True to work with dense arrays/tensors
        Returns:
                K_ij[LazyTensor] if dense = False
                K_ij[np.array or torch.tensor] if dense = True
        """

        if y is None:
            y = x
        x = x / self.sigma
        y = y / self.sigma

        x_i, x_j = (
            self.tools.contiguous(self._to_device(x[:, None, :])),
            self.tools.contiguous(self._to_device(y[None, :, :])),
        )

        if self.kernel == "rbf":
            if dense:
                D_ij = ((x_i - x_j) ** 2).sum(axis=2)
                K_ij = self.tools.exp(-D_ij)

            else:
                x_i, x_j = self.LazyTensor(x_i), self.LazyTensor(x_j)
                D_ij = ((x_i - x_j) ** 2).sum(dim=2)
                K_ij = (-D_ij).exp()

                # block-sparse reduction preprocess
                K_ij = self._Gauss_block_sparse_pre(x, y, K_ij)
        elif self.kernel == "exp":
            if dense:
                K_ij = self.tools.exp(
                    -self.tools.sqrt((((x_i - x_j) ** 2).sum(axis=2)))
                )

            else:
                x_i, x_j = self.LazyTensor(x_i), self.LazyTensor(x_j)
                K_ij = (-(((x_i - x_j) ** 2).sum(-1)).sqrt()).exp()

                # block-sparse reduction preprocess
                K_ij = self._Gauss_block_sparse_pre(x, y, K_ij)

        # computation with custom kernel
        else:
            print("Please note that computations on custom kernels are dense-only.")
            K_ij = self.kernel(x_i, x_j)

        return K_ij

    def _Gauss_block_sparse_pre(
        self, x: generic_array, y: generic_array, K_ij: GenericLazyTensor
    ):
        """
        Helper function to preprocess data for block-sparse reduction
        of the Gaussian kernel
        Args:
            x, y =  arrays or tensors giving rise to Gaussian kernel K(x,y)
            K_ij = symbolic representation of K(x,y)
            eps[float] = size for square bins
        Returns:
            K_ij =  symbolic representation of K(x,y) with
                                set sparse ranges
        """
        x = self._to_device(x)
        y = self._to_device(y)
        # labels for low dimensions
        if x.shape[1] < 4 or y.shape[1] < 4:
            x_labels = self.tools.grid_cluster(x, self.eps)
            y_labels = self.tools.grid_cluster(y, self.eps)

            # range and centroid per class
            x_ranges, x_centroids, _ = self.tools.cluster_ranges_centroids(x, x_labels)
            y_ranges, y_centroids, _ = self.tools.cluster_ranges_centroids(y, y_labels)

        else:
            # labels for higher dimensions
            x_labels, x_centroids = self._KMeans(x)
            y_labels, y_centroids = self._KMeans(y)
            # compute ranges
            x_ranges = self.tools.cluster_ranges(x_labels)
            y_ranges = self.tools.cluster_ranges(y_labels)

        # sort points
        x, x_labels = self.tools.sort_clusters(x, x_labels)
        y, y_labels = self.tools.sort_clusters(y, y_labels)

        # Compute a coarse Boolean mask:
        if self.kernel == "rbf":
            D = self.tools.arraysum(
                (x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2, 2
            )

        elif self.kernel == "exp":
            D = self.tools.sqrt(
                self.tools.arraysum(
                    (x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2, 2
                )
            )

        keep = D < (self.mask_radius) ** 2
        # mask -> set of integer tensors
        ranges_ij = self.tools.from_matrix(x_ranges, y_ranges, keep)
        K_ij.ranges = ranges_ij  # block-sparsity pattern

        return K_ij

    def _astype(self, data, type):
        return data

    def _to_device(self, data):
        return data

    def _update_dtype(self, x):
        """Helper function that sets dtype to that of
            the given data in the fitting step.
        Args:
            x [np.array or torch.tensor] = raw data to remap
        Returns:
            None
        """
        self.dtype = x.dtype
        self.inv_eps = np.array([self.inv_eps]).astype(self.dtype)[0]

    def _check_random_state(self, seed: Union[None, int]) -> None:
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

    def _KMeans(self, x: generic_array) -> Tuple[generic_array]:
        """K-means algorithm to find clusters for preprocessing"""

        raise NotImplementedError("Subclass must implement this method.")
