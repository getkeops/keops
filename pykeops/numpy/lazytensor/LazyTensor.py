import numpy as np

from pykeops.common.lazy_tensor import GenericLazyTensor, ComplexGenericLazyTensor
from pykeops.numpy.utils import numpytools

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
    Simple wrapper that returns an instantiation of :class:`LazyTensor` of type 0.
    """
    return Var(x_or_ind, dim, 0)


def Vj(x_or_ind, dim=None):
    r"""
    Simple wrapper that returns an instantiation of :class:`LazyTensor` of type 1.
    """
    return Var(x_or_ind, dim, 1)


def Pm(x_or_ind, dim=None):
    r"""
    Simple wrapper that returns an instantiation of :class:`LazyTensor` of type 2.
    """
    return Var(x_or_ind, dim, 2)


class LazyTensor(GenericLazyTensor):
    r"""Symbolic wrapper for NumPy arrays.

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

    def __new__(self, x=None, axis=None, is_complex=False):
        if is_complex or numpytools.detect_complex(x):
            return ComplexLazyTensor(x, axis)
        else:
            return object.__new__(self)

    def __init__(self, x=None, axis=None, is_complex=False):
        super().__init__(x=x, axis=axis)

    def get_tools(self):
        self.tools = numpytools
        self.Genred = numpytools.Genred
        self.KernelSolve = numpytools.KernelSolve

    def lt_constructor(self, x=None, axis=None, is_complex=False):
        return LazyTensor(x=x, axis=axis, is_complex=is_complex)


class ComplexLazyTensor(ComplexGenericLazyTensor):
    r"""Extension of the LazyTensor class for complex operations."""

    def __init__(self, x=None, axis=None):
        super().__init__(x=x, axis=axis)

    def get_tools(self):
        self.tools = numpytools
        self.Genred = numpytools.Genred
        self.KernelSolve = numpytools.KernelSolve

    def lt_constructor(self, x=None, axis=None, is_complex=True):
        return LazyTensor(x=x, axis=axis, is_complex=is_complex)
