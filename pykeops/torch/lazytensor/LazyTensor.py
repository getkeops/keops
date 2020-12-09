import torch

from pykeops.common.lazy_tensor import GenericLazyTensor
from pykeops.torch.utils import torchtools


# Convenient aliases:


def Var(x_or_ind, dim=None, cat=None):
    if dim is None:
        # init via data: we assume x_or_ind is data
        return LazyTensor(x_or_ind, axis=cat)
    else:
        # init via symbolic variable given as triplet (ind,dim,cat)
        return LazyTensor((x_or_ind, dim, cat))


def Vi(x_or_ind, dim=None):
    r"""
    Simple wrapper that return an instantiation of :class:`LazyTensor` of type 0.
    """
    return Var(x_or_ind, dim, 0)


def Vj(x_or_ind, dim=None):
    r"""
    Simple wrapper that return an instantiation of :class:`LazyTensor` of type 1.
    """
    return Var(x_or_ind, dim, 1)


def Pm(x_or_ind, dim=None):
    r"""
    Simple wrapper that return an instantiation of :class:`LazyTensor` of type 2.
    """
    return Var(x_or_ind, dim, 2)


class LazyTensor(GenericLazyTensor):
    r"""Symbolic wrapper for PyTorch tensors.

    :class:`LazyTensor` encode numerical arrays through the combination
    of a symbolic, **mathematical formula** and a list of **small data arrays**.
    They can be used to implement efficient algorithms on objects
    that are **easy to define**, but **impossible to store** in memory
    (e.g. the matrix of pairwise distances between
    two large point clouds).

    :class:`LazyTensor` may be created from standard NumPy arrays or PyTorch tensors,
    combined using simple mathematical operations and converted
    back to NumPy arrays or PyTorch tensors with
    efficient reduction routines, which outperform
    standard tensorized implementations by two orders of magnitude.
    """

    def __init__(self, x=None, axis=None):
        super().__init__(x=x, axis=axis)

        # torch specialization
        typex = type(x)

        if typex not in [type(None), tuple, int, list, torch.Tensor] + self.float_types:
            raise TypeError(
                "LazyTensors should be built from PyTorch tensors, "
                "float/integer numbers, lists of floats or 3-uples of integers. "
                "Received: {}".format(typex)
            )

        if typex == torch.Tensor and len(x.shape) == 0:  # Torch scalar -> Torch tensor
            x = x.view(1)
        elif typex in self.float_types:
            x = torch.Tensor([x]).view(1)

        if typex == torch.Tensor:
            self.infer_dim(x, axis)

    def get_tools(self):
        self.tools = torchtools
        self.Genred = torchtools.Genred
        self.KernelSolve = torchtools.KernelSolve

    def lt_constructor(self, x=None, axis=None):
        return LazyTensor(x=x, axis=axis)

    float_types = [float]
