import torch
from .. import config as pykeopsconfig

##########################################################
# Check Pytorch install

# is the proper torch version  installed ?
torch_version_required = "1.3"

if torch.__version__ < torch_version_required:
    raise ImportError(
        "[pyKeOps]: The pytorch version should be >=" + torch_version_required
    )

##########################################################
# Get GPU informations

pykeopsconfig.gpu_available = torch.cuda.is_available()  # use torch to detect gpu

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
from .lazytensor.LazyTensor import LazyTensor, ComplexLazyTensor, Vi, Vj, Pm

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
