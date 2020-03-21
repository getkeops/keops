import torch

import pykeops

##########################################################
# Check Pytorch install

# is the proper torch version  installed ?
torch_version_required = '1.3'

if torch.__version__ < torch_version_required:
    raise ImportError('The pytorch version should be >=' + torch_version_required)

# get some useful informations about pytorch ABI
from torch.utils.cpp_extension import include_paths

include_dirs = include_paths()[0:2]
torch_cxx11_abi_flag = torch._C._GLIBCXX_USE_CXX11_ABI

##########################################################
# Get GPU informations

pykeops.gpu_available = torch.cuda.is_available() # use torch to detect gpu
pykeops.torch_found = True

default_dtype = 'float32'

##########################################################
# Import pyKeOps routines 

from .generic.generic_red import Genred
from .operations import KernelSolve
from .kernel_product.kernels import Kernel, kernel_product, kernel_formulas
from .generic.generic_ops import generic_sum, generic_logsumexp, generic_argmin, generic_argkmin
from .kernel_product.formula import Formula
from pykeops.common.lazy_tensor import LazyTensor, Vi, Vj, Pm

# N.B.: If "from pykeops.numpy import LazyTensor" has already been run,
#       the line above will *not* import "torchtools" and we'll end up with an error...
#       So even though it may be a bit ugly, we re-load the lazy_tensor file
#       to make sure that "pykeops.torch_found = True" is taken into account.
import importlib

importlib.reload(pykeops.common.lazy_tensor)

__all__ = sorted(
    ["Genred", "generic_sum", "generic_logsumexp", "generic_argmin", "generic_argkmin", "Kernel", "kernel_product",
     "KernelSolve", "kernel_formulas", "Formula", "LazyTensor", "Vi", "Vj", "Pm"])
