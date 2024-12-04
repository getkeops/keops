from .cpu import *
from ..config import get_cuda_config

cuda_config = get_cuda_config()

if cuda_config._use_cuda:
    from .gpu import *
