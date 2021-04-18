from pykeops.common.ivf import GenericIVF
import torch


class IVF(GenericIVF):
    """IVF-Flat is a KNN approximation algorithm that first clusters the data and then performs the query search on a subset of the input dataset."""

    def __init__(self, k=5, metric="euclidean", normalise=False):
        """Initialise the IVF-Flat class.

        IVF-Flat is a KNN approximation algorithm that first clusters the data and then performs the query search on a subset of the input dataset.

        Args:
          k (int): Number of nearest neighbours to obtain
          metric (str,function): Metric to use
            Currently, "euclidean", "manhattan", "angular" and "hyperbolic" are directly supported, apart from custom metrics
            Hyperbolic metric requires the use of approx = True, during the fit() function later
            Custom metrics should be in the form of a function with 2 inputs and returns their distance
            For more information, refer to the tutorial
          normalise (bool): Whether or not to normalise all input data to norm 1
            This is used mainly for angular metric
            In place of this, "angular_full" metric may be used instead

        """
        from pykeops.torch import LazyTensor

        self.__get_tools()
        super().__init__(k=k, metric=metric, normalise=normalise, lazytensor=LazyTensor)

    def __get_tools(self):
        from pykeops.torch.utils import torchtools

        self.tools = torchtools

    def fit(self, x, clusters=50, a=5, Niter=15, approx=False, n=50):
        """Fits a dataset to perform the nearest neighbour search over

        K-Means is performed on the dataset to obtain clusters
        Then the closest clusters to each cluster is stored for use during query time

        Args:
          x (torch.Tensor): Torch tensor dataset of shape N, D
            Where N is the number of points and D is the number of dimensions
          clusters (int): Total number of clusters to create in K-Means
          a (int): Number of clusters to search over, must be less than total number of clusters created
          Niter (int): Number of iterations to run in K-Means algorithm
          approx (bool): Whether or not to use an approximation step in K-Means
            In hyperbolic metric and custom metric, this should be set to True
            This is because the optimal cluster centroid may not have a simple closed form expression
          n (int): Number of iterations to optimise the cluster centroid, when approx = True
            A value of around 50 is recommended
            Lower values are faster while higher values give better accuracy in centroid location

        """
        return self._fit(
            x, clusters=clusters, a=a, Niter=Niter, device=x.device, approx=approx, n=n
        )

    def kneighbors(self, y):
        """Obtains the nearest neighbors for an input dataset from the fitted dataset

        Args:
          y (torch.Tensor): Input dataset to search over
        """
        return self._kneighbors(y)
