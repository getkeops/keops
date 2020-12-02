"""
KeOps specific
========================================
"""

#########################
# (N.B.: with data on device would be slightly better!)
#


import torch
import numpy as np
from time import time

######################################################################
# Benchmark specifications:
#

nits = 1
Ns, D = [10000, 100000, 1000000], 3

dtype = "float32"

from pykeops.numpy import RadialKernelConv

my_conv = RadialKernelConv(dtype)


def KP(x, y, p):
    return my_conv(x, y, p, 1.0, kernel="gaussian")


for N in Ns:

    # Generate the data
    x = np.random.randn(N, D).astype(dtype)
    y = np.random.randn(N, D).astype(dtype)
    p = np.random.randn(N, 1).astype(dtype)

    # First run just in case...
    p = KP(x, y, p)

    # Timings for KeOps specific
    start = time()
    for _ in range(nits):
        p = KP(x, y, p)

    end = time()
    print("Timing with {} points: {} x {:.4f}s".format(N, nits, (end - start) / nits))
