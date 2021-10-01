from keops.mapreduce.cpu import *
from keops.config.config import use_cuda
if use_cuda:
    from keops.mapreduce.gpu import *

