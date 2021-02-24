# !pip install pykeops[full] > install.log
# colab for this code
# https://colab.research.google.com/drive/1vF2cOSddbRFM5PLqxkIzyZ9XkuzO5DKN?usp=sharing
import numpy as np
import torch
import pykeops

from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import from_matrix
from pykeops.torch.cluster import cluster_ranges_centroids, cluster_ranges
from pykeops.torch.cluster import sort_clusters
from pykeops.torch import LazyTensor

from sklearn.utils import check_random_state

from scipy.sparse.linalg import aslinearoperator, eigsh
from scipy.sparse.linalg.interface import IdentityOperator


################################################################################
# Same as LazyNystrom_T but written with pyKeOps
import numpy as np
import torch
import pykeops

from pykeops.numpy import LazyTensor as LazyTensor_n
from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import from_matrix
from pykeops.torch.cluster import cluster_ranges_centroids, cluster_ranges
from pykeops.torch.cluster import sort_clusters
from pykeops.torch import LazyTensor

from sklearn.utils import check_random_state, as_float_array
from scipy.linalg import svd

from scipy.sparse.linalg import aslinearoperator, eigsh
from scipy.sparse.linalg.interface import IdentityOperator
from pykeops.torch import Genred

import matplotlib.pyplot as plt
import time


################################################################################
# Same as LazyNystrom_T but written with pyKeOps

class LazyNystrom_TK:
    '''
        Class to implement Nystrom on torch LazyTensors.
        This class works as an interface between lazy tensors and
        the Nystrom algorithm in NumPy.
        * The fit method computes K^{-1}_q.
        * The transform method maps the data into the feature space underlying
        the Nystrom-approximated kernel.
        * The method K_approx directly computes the Nystrom approximation.
        Parameters:
        n_components [int] = how many samples to select from data.
        kernel [str] = type of kernel to use. Current options = {linear, rbf}.
        gamma [float] = exponential constant for the RBF kernel.
        random_state=[None, float] = to set a random seed for the random
                                     sampling of the samples. To be used when
                                     reproducibility is needed.
    '''

    def __init__(self, n_components=100, kernel='rbf', sigma: float = 1.,
                 exp_sigma: float = 1.0, eps: float = 0.05, mask_radius: float = None,
                 k_means=10, n_iter: int = 10, inv_eps: float = None, dtype=np.float32,
                 backend='CPU', random_state=None):

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
        self.backend = backend  # conditional here
        if inv_eps:
            self.inv_eps = inv_eps
        else:
            if kernel == 'linear':
                self.inv_eps = 1e-4
            else:
                self.inv_eps = 1e-8
        if not mask_radius:
            if kernel == 'rbf':
                self.mask_radius = 2 * np.sqrt(2) * self.sigma
            if kernel == 'exp':
                self.mask_radius = 8 * self.exp_sigma

    def fit(self, X: torch.tensor):
        '''
        Args:   X = torch tensor with features of shape
                (1, n_samples, n_features)
        Returns: Fitted instance of the class
        '''

        # Basic checks: we have a lazy tensor and n_components isn't too large
        assert type(X) == torch.Tensor, 'Input to fit(.) must be a Tensor.'
        assert X.size(0) >= self.n_components, f'The application needs X.shape[1] >= n_components.'
        # self._update_dtype(X)
        # Number of samples
        n_samples = X.size(0)
        # Define basis
        rnd = check_random_state(self.random_state)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:self.n_components]
        basis = X[basis_inds]
        # Build smaller kernel
        basis_kernel = self._pairwise_kernels(basis, kernel=self.kernel)
        # Get SVD
        U, S, V = torch.svd(basis_kernel)
        S = torch.maximum(S, torch.ones(S.size()) * 1e-12)
        self.normalization_ = torch.mm(U / np.sqrt(S), V.t())
        self.components_ = basis
        self.component_indices_ = inds

        return self

    def transform(self, X: torch.tensor) -> torch.tensor:
        ''' Applies transform on the data.
        Args:
            X [LazyTensor] = data to transform
        Returns
            X [LazyTensor] = data after transformation
        '''
        K_nq = self._pairwise_kernels(X, self.components_, self.kernel)
        return K_nq @ self.normalization_.t()

    def K_approx(self, X: torch.tensor) -> torch.tensor:
        ''' Function to return Nystrom approximation to the kernel.
        Args:
            X[torch.tensor] = data used in fit(.) function.
        Returns
            K[torch.tensor] = Nystrom approximation to kernel'''

        K_nq = self._pairwise_kernels(X, self.components_, self.kernel)
        K_approx = K_nq @ self.normalization_ @ K_nq.t()
        return K_approx

    def _pairwise_kernels(self, x: torch.tensor, y: torch.tensor = None, kernel='rbf',
                          sigma: float = 1.) -> LazyTensor:
        '''Helper function to build kernel
        Args:   X = torch tensor of dimension 2.
                K_type = type of Kernel to return
        Returns:
                K_ij[LazyTensor]
        '''
        if y is None:
            y = x
        if kernel == 'linear':
            K_ij = x @ y.T
        elif kernel == 'rbf':
            x /= sigma
            y /= sigma

            x_i, x_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
            K_ij = (-1 * ((x_i - x_j) ** 2).sum(-1)).exp()

            # block-sparse reduction preprocess
            K_ij = self._Gauss_block_sparse_pre(x, y, K_ij)


        elif kernel == 'exp':
            x_i, x_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
            K_ij = (-1 * ((x_i - x_j) ** 2).sum().sqrt()).exp()
            # block-sparse reduction preprocess
            K_ij = self._Gauss_block_sparse_pre(x, y, K_ij)  # TODO

        K_ij = K_ij @ torch.diag(torch.ones(K_ij.shape[1]))  # make 1 on diag only

        K_ij.backend = self.backend
        return K_ij

    def _Gauss_block_sparse_pre(self, x: torch.tensor, y: torch.tensor, K_ij: LazyTensor):
        '''
        Helper function to preprocess data for block-sparse reduction
        of the Gaussian kernel

        Args:
            x[np.array], y[np.array] = arrays giving rise to Gaussian kernel K(x,y)
            K_ij[LazyTensor_n] = symbolic representation of K(x,y)
            eps[float] = size for square bins
        Returns:
            K_ij[LazyTensor_n] = symbolic representation of K(x,y) with
                                set sparse ranges
        '''
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
        D = torch.sum((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2, 2)
        keep = D < (self.mask_radius) ** 2
        # mask -> set of integer tensors
        ranges_ij = from_matrix(x_ranges, y_ranges, keep)
        K_ij.ranges = ranges_ij  # block-sparsity pattern

        return K_ij

    def _KMeans(self, x: torch.tensor):
        ''' KMeans with Pykeops to do binning of original data.
        Args:
            x[np.array] = data
            k_means[int] = number of bins to build
            n_iter[int] = number iterations of KMeans loop
        Returns:
            labels[np.array] = class labels for each point in x
            clusters[np.array] = coordinates for each centroid
        '''

        N, D = x.shape
        clusters = torch.clone(x[:self.k_means, :])  # initialization of clusters
        x_i = LazyTensor(x[:, None, :])

        for i in range(self.n_iter):

            clusters_j = LazyTensor(clusters[None, :, :])
            D_ij = ((x_i - clusters_j) ** 2).sum(-1)  # points-clusters kernel
            labels = D_ij.argmin(axis=1).reshape(N)  # Points -> Nearest cluster
            Ncl = torch.bincount(labels)  # Class weights
            for d in range(D):  # Compute the cluster centroids with np.bincount:
                clusters[:, d] = torch.bincount(labels, weights=x[:, d]) / Ncl

        return labels, clusters

    def _update_dtype(self, x):
        ''' Helper function that sets inv_eps to dtype to that of
            the given data in the fitting step.

        Args:
            x [np.array] = raw data to remap
        Returns:
            nothing
        '''
        self.dtype = x.dtype
        self.inv_eps = np.array([self.inv_eps]).astype(self.dtype)[0]