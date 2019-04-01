import pykeops

##########################################################
# Search for Pytorch and a GPU

torch_version_required = '1.0'

# is torch installed ?
import torch
from torch.utils.cpp_extension import include_paths

include_dirs = include_paths()[0:2]

if torch.__version__ < torch_version_required:
    raise ImportError('The pytorch version should be >=' + torch_version_required)

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

__all__ = sorted(["Genred", "generic_sum", "generic_logsumexp", "generic_argmin", "generic_argkmin", "Kernel", "kernel_product", "KernelSolve", "kernel_formulas", "Formula"])
