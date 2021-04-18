from pykeops.common.ivf import GenericIVF
import numpy as np


class IVF(GenericIVF):
    """IVF-Flat is a KNN approximation algorithm that first clusters the data and then performs the query search on a subset of the input dataset."""

    def __init__(self, k=5, metric="euclidean", normalise=False):
        """Initialise the IVF-Flat class.

        IVF-Flat is a KNN approximation algorithm that first clusters the data and then performs the query search on a subset of the input dataset.

        Args:
          k (int): Number of nearest neighbours to obtain
          metric (str,function): Metric to use
            Currently, "euclidean", "manhattan" and "angular" are directly supported
            Custom metrics are not supported in numpy, please use torch version instead
            For more information, refer to the tutorial
          normalise (bool): Whether or not to normalise all input data to norm 1
            This is used mainly for angular metric
            In place of this, "angular_full" metric may be used instead

        """
        from pykeops.numpy import LazyTensor

        self.__get_tools()
        super().__init__(k=k, metric=metric, normalise=normalise, lazytensor=LazyTensor)

    def __get_tools(self):
        from pykeops.numpy.utils import numpytools

        self.tools = numpytools

    def fit(self, x, clusters=50, a=5, Niter=15, backend="CPU", approx=False):
        """Fits a dataset to perform the nearest neighbour search over

        K-Means is performed on the dataset to obtain clusters
        Then the closest clusters to each cluster is stored for use during query time

        Args:
          x (torch.Tensor): Torch tensor dataset of shape N, D
            Where N is the number of points and D is the number of dimensions
          clusters (int): Total number of clusters to create in K-Means
          a (int): Number of clusters to search over, must be less than total number of clusters created
          Niter (int): Number of iterations to run in K-Means algorithm

        """
        if approx:
            raise ValueError("Approximation not supported for numpy")
        return self._fit(x, clusters=clusters, a=a, Niter=Niter, backend=backend)

    def kneighbors(self, y):
        """Obtains the nearest neighbors for an input dataset from the fitted dataset

        Args:
          y (np.ndarray): Input dataset to search over
        """
        return self._kneighbors(y)
