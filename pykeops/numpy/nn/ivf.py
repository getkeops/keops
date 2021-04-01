from pykeops.numpy import LazyTensor
from pykeops.numpy.cluster import cluster_ranges_centroids
from pykeops.numpy.cluster import from_matrix
from pykeops.common.ivf import GenericIVF
from pykeops.numpy.utils import numpytools

import numpy as np


class IVF(GenericIVF):
    def __init__(self, k=5, metric="euclidean", normalise=False):
        self.__get_tools()
        super().__init__(k=k, metric=metric, normalise=normalise, LazyTensor=LazyTensor)

    def __get_tools(self):
        self.tools = numpytools

    def fit(self, x, clusters=50, a=5, Niter=15, backend="CPU"):
        if type(x) != np.ndarray:
            raise ValueError("Input dataset must be a np array")
        return self._fit(x, clusters=clusters, a=a, Niter=Niter, backend=backend)

    def kneighbors(self, y):
        if type(y) != np.ndarray:
            raise ValueError("Query dataset must be a np array")
        return self._kneighbors(y)
