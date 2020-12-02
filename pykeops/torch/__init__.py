import torch
import pykeops.config

##########################################################
# Check Pytorch install

# is the proper torch version  installed ?
torch_version_required = "1.3"

if torch.__version__ < torch_version_required:
    raise ImportError(
        "[pyKeOps]: The pytorch version should be >=" + torch_version_required
    )

# get the path of the current pytorch and some built options
include_dirs = [
    "-DPYTORCH_ROOT_DIR=" + ";".join(torch.__path__),
    "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
]

##########################################################
# Get GPU informations

pykeops.config.gpu_available = torch.cuda.is_available()  # use torch to detect gpu
default_dtype = "float32"

##########################################################
# Import pyKeOps routines

from .generic.generic_red import Genred
from .generic.generic_ops import (
    generic_sum,
    generic_logsumexp,
    generic_argmin,
    generic_argkmin,
)
from .operations import KernelSolve
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
