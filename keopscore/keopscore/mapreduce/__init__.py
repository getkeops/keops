from keopscore.config import cuda

from .cpu import *
if cuda.get_use_cuda():
    from .gpu import *
