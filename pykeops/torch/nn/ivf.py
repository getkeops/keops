from pykeops.torch import LazyTensor
from pykeops.common.ivf import GenericIVF
from pykeops.torch.utils import torchtools
import torch


class IVF(GenericIVF):
    def __init__(self, k=5, metric="euclidean", normalise=False):
        self.__get_tools()
        super().__init__(k=k, metric=metric, normalise=normalise, LazyTensor=LazyTensor)

    def __get_tools(self):
        self.tools = torchtools

    def fit(self, x, clusters=50, a=5, Niter=15, approx=False, n=50):
        if type(x) != torch.Tensor:
            raise ValueError("Input dataset must be a torch tensor")
        return self._fit(
            x, clusters=clusters, a=a, Niter=Niter, device=x.device, approx=approx, n=n
        )

    def kneighbors(self, y):
        if type(y) != torch.Tensor:
            raise ValueError("Query dataset must be a torch tensor")
        return self._kneighbors(y)
