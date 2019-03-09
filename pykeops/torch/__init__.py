import pykeops

##########################################################
# Search for Pytorch and a GPU

torch_version_required = '0.4.1'

# is torch installed ?
import torch
from torch.utils import cpp_extension

if torch.__version__ < torch_version_required:
    raise ImportError('The pytorch version should be ==' + torch_version_required)

pykeops.gpu_available = torch.cuda.is_available() # use torch to detect gpu
pykeops.torch_found = True

default_cuda_type = 'float32'

##########################################################
# Import pyKeOps routines 

from .operations import Genred
from .kernel_product.kernels import Kernel, kernel_product, kernel_formulas
from .generic.generic_ops import generic_sum, generic_logsumexp, generic_argmin, generic_argkmin
from .kernel_product.formula import Formula
