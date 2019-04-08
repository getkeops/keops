import pykeops

##########################################################
# Search for GPU

import GPUtil

try:
    pykeops.gpu_available = len(GPUtil.getGPUs()) > 0
except:
    pykeops.gpu_available = False

default_dtype = 'float64' # 'float32' or 'float64'

##########################################################
# Import pyKeOps routines 

from .generic.generic_red import Genred
from .operations import KernelSolve
from .convolutions.radial_kernel import RadialKernelConv, RadialKernelGrad1conv
from .generic.generic_ops import generic_sum, generic_logsumexp, generic_argmin, generic_argkmin

__all__ = sorted(["Genred", "generic_sum", "generic_logsumexp", "generic_argmin", "generic_argkmin", "KernelSolve"])


