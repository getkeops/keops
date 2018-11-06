import pykeops

##########################################################
# Search for GPU

import GPUtil

try:
    pykeops.gpu_available = len(GPUtil.getGPUs()) > 0
except:
    pykeops.gpu_available = False


##########################################################
# Import pyKeOps routines 

from pykeops.numpy.generic.generic_red import Genred
from .convolutions.radial_kernel import RadialKernelConv, RadialKernelGrad1conv

