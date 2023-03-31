from .cpu import *
from ..config.config import use_cuda

if use_cuda:
    from .gpu import *
