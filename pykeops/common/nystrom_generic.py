import numpy as np
import pykeops
from typing import TypeVar, Union
import warnings

# Generic placeholder for numpy and torch variables.
generic_array = TypeVar("generic_array")
GenericLazyTensor = TypeVar("GenericLazyTensor")


class GenericNystroem:
    """
    Super class defining the Nystrom operations. The end user should
    use numpy.nystrom or torch.nystrom subclasses.

    """

    def __init__(
        self,
        n_components: int = 100,
        kernel: Union[str, callable] = "rbf",
        sigma: float = None,
        inv_eps: float = None,
        verbose: bool = False,
        random_state: Union[None, int] = None,
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
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma
        self.dtype = None
        self.verbose = verbose
        self.random_state = random_state
        self.tools = None
        self.lazy_tensor = None

        if inv_eps:
            self.inv_eps = inv_eps
        else:
            self.inv_eps = 1e-8

    def fit(self, x: generic_array) -> "GenericNystroem":
        """
        Args:
            x: generic_array: array or tensor of shape (n_samples, n_features)
        Returns:
            Fitted instance of the class
        """
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
        if self.sigma is None:
            self.sigma = np.sqrt(x.shape[1]) / np.sqrt(2)

        # Update dtype
        self._update_dtype()
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
        self.normalization = self._decomposition_and_norm(basis_kernel)
        self.components_ = basis
        self.component_indices_ = inds

        return self

    def _decomposition_and_norm(self, X: GenericLazyTensor):
        """
        To be defined in the subclass

        Args:
          X: GenericLazyTensor:
        """
        raise NotImplementedError(
            "Subclass must implement the method _decomposition_and_norm."
        )

    def _get_kernel(self, x: generic_array, y: generic_array) -> generic_array:
        """
        To be implmented in the subclass.

        Args:
            x: generic_array
            y: generic_array

        Returns:
            K: generic_array: dense kernel array

        """
        raise NotImplementedError("Subclass must implement the method _get_kernel.")

    def transform(self, x: generic_array, dense=True) -> generic_array:
        """Applies transform on the data mapping it to the feature space
        which supports the approximated kernel.

        Args:
            X: generic_array: data to transform, dim: n_samples n x m
        Returns
            X(array): data after transformation, dim: n_samples n x n_components D
            x:generic_array:
          dense:  (Default value = True)

        Returns:

        """
        if type(x) == np.ndarray and not dense:
            warnings.warn("For Numpy transform it is best to use dense=True")

        K_nq = self._pairwise_kernels(x, self.components_, dense=dense)
        x_new = K_nq @ self.normalization
        return x_new

    def _pairwise_kernels(self, x: generic_array, y: generic_array = None, dense=False):
        """Helper function to build kernel

                y(np.array or torch.tensor): array/tensor N x D
                dense(bool): False to work with lazy tensor reduction,
                              True to work with dense arrays/tensors

        Args:
          x:generic_array: data, shape N x M
          y:generic_array:  (Default value = None), if given N x D array
          dense: boolean: (Default value = False). Use False to return a
            to return a dense generic_array. Use True to return a LazyTensor
            version.

        Returns:
          LazyTensor: if dense == False
          dense array: if dense == True

        """

        if y is None:
            y = x
        x = x / (np.sqrt(2) * self.sigma)
        y = y / (np.sqrt(2) * self.sigma)

        x_i, x_j = self.tools.contiguous(x[:, None, :]), self.tools.contiguous(
            y[None, :, :]
        )  # (N, 1, M), (1, N, M) or (1, N, D)

        if self.kernel == "rbf":
            if dense:
                K_ij = self._get_kernel(x, y)

            else:
                x_i, x_j = self.lazy_tensor(x_i), self.lazy_tensor(x_j)
                D_ij = ((x_i - x_j) ** 2).sum(dim=2)
                K_ij = (-D_ij).exp()

        elif self.kernel == "exp":
            if dense:
                K_ij = self._get_kernel(x, y, kernel="exp")

            else:
                x_i, x_j = self.lazy_tensor(x_i), self.lazy_tensor(x_j)
                K_ij = (-(((x_i - x_j) ** 2).sum(-1)).sqrt()).exp()

        # computation with custom kernel
        else:
            print("Please note that computations on custom kernels are dense-only.")
            K_ij = self.kernel(x_i, x_j)

        return K_ij  # (N, N)

    def _update_dtype(self) -> None:
        """Helper function that sets dtype to that of
        the given data in the fitting step. Fixes inv_eps data type to
        that of input data.
        """
        self.inv_eps = np.array([self.inv_eps]).astype(self.dtype)[0]

    def _check_random_state(self, seed: Union[None, int]) -> None:
        """
        Set/get np.random.RandomState instance for permutation.

        Args:
            seed: Union[None: int]:

        Returns:
            numpy random state
        """

        if seed is None:
            return np.random.mtrand._rand

        elif type(seed) == int:
            return np.random.RandomState(seed)

        raise ValueError(f"Seed {seed} must be None or an integer.")
