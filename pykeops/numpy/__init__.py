import pykeops

##########################################################
# Search for GPU

import GPUtil

try:
    pykeops.gpu_available = len(GPUtil.getGPUs()) > 0
except:
    pykeops.gpu_available = False

default_cuda_type = 'float64' # 'float32' or 'float64'

##########################################################
# Import pyKeOps routines 

from .operations import Genred
from .convolutions.radial_kernel import RadialKernelConv, RadialKernelGrad1conv

