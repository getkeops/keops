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

pykeops.numpy_found = True

from .generic.generic_red import Genred
from .operations import KernelSolve
from .convolutions.radial_kernel import RadialKernelConv, RadialKernelGrad1conv
from .generic.generic_ops import generic_sum, generic_logsumexp, generic_argmin, generic_argkmin
from pykeops.common.lazy_tensor import LazyTensor, Vi, Vj, Pm

# N.B.: If "from pykeops.torch import LazyTensor" has already been run,
#       the line above will *not* import "numpytools" and we'll end up with an error...
#       So even though it may be a bit ugly, we re-load the lazy_tensor file
#       to make sure that "pykeops.numpy_found = True" is taken into account.
import importlib
importlib.reload(pykeops.common.lazy_tensor)


__all__ = sorted(["Genred", "generic_sum", "generic_logsumexp", "generic_argmin", "generic_argkmin", "KernelSolve", "LazyTensor", "Vi", "Vj", "Pm"])


