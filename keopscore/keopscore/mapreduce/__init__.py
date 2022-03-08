from keopscore.mapreduce.cpu import *
from keopscore.config.config import use_cuda

if use_cuda:
    from keopscore.mapreduce.gpu import *
