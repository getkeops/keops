import pykeops.config

default_dtype = "float64"  # float32' or 'float64'

##########################################################
# Import pyKeOps routines


from .generic.generic_red import Genred
from .operations import KernelSolve
from .convolutions.radial_kernel import RadialKernelConv, RadialKernelGrad1conv
from .generic.generic_ops import (
    generic_sum,
    generic_logsumexp,
    generic_argmin,
    generic_argkmin,
)
from .lazytensor.LazyTensor import LazyTensor, Vi, Vj, Pm

__all__ = sorted(
    [
        "Genred",
        "generic_sum",
        "generic_logsumexp",
        "generic_argmin",
        "generic_argkmin",
        "KernelSolve",
        "LazyTensor",
        "Vi",
        "Vj",
        "Pm",
    ]
)
