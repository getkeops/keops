#####################################################
###  specialized classes for PyTorch backend
#####################################################

from pykeops.torch import Genred
from pykeops.symbolictensor.genericsymbolictensor import (
    GenericSymbolicTensor,
    GenericVariable,
)
from pykeops.symbolictensor.operations import Op


class PytorchSymbolicTensor(GenericSymbolicTensor):
    Genred = Genred

    @property
    def SymbolicTensor_backend(self):
        return PytorchSymbolicTensor

    @staticmethod
    def SymbolicTensorConstructor(node=Op(), children=()):
        return PytorchSymbolicTensor(node=node, children=children)


class Variable(GenericVariable, PytorchSymbolicTensor):
    def __init__(self, tensor, nbatchdims):
        super().__init__(tensor, nbatchdims)


def SymbolicTensor(tensor, nbatchdims=0):
    return Variable(tensor, nbatchdims)


#####################################################
###  decorator
#####################################################


def keops(fun):
    def newfun(*args, **kwargs):
        newargs = [SymbolicTensor(arg) for arg in args]
        res = fun(*newargs, **kwargs)
        return res.dense()

    return newfun
