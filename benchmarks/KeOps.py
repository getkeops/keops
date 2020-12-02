"""
KeOps
=====
"""

import torch
import numpy as np
from time import time

nits = 10
Ns, D = [10000, 100000, 1000000], 3

from pykeops.torch import generic_sum

KP = generic_sum(
    "Exp(-SqDist(X,Y)) * B",  # Formula
    "A = Vi(1)",  # Output
    "X = Vi({})".format(D),  # 1st argument
    "Y = Vj({})".format(D),  # 2nd argument
    "B = Vj(1)",
)  # 3rd argument

for N in Ns:

    # Generate the data
    x = torch.randn(N, D).cuda()
    y = torch.randn(N, D).cuda()
    p = torch.randn(N, 1).cuda()

    # First run just in case...
    p = KP(x, y, p)

    # Timings for KeOps
    start = time()
    for _ in range(nits):
        p = KP(x, y, p)

    end = time()
    print("Timing with {} points: {} x {:.4f}s".format(N, nits, (end - start) / nits))
