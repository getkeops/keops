"""
PyTorch, on the GPU
===========================
"""

####################################
# Blabla
#


import torch
import numpy as np
from time import time


nits = 100
Ns, D = [10000, 100000, 1000000], 3


def KP(x, y, b):
    D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
    D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
    D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
    D_xy = D_xx - 2 * D_xy + D_yy
    K_xy = (-D_xy).exp()

    return K_xy @ b


for N in Ns:

    # Generate the data
    x = torch.randn(N, D).cuda()
    y = torch.randn(N, D).cuda()
    p = torch.randn(N, 1).cuda()

    # First run to warm-up the device...
    p = KP(x, y, p)

    # Actual timings:
    start = time()
    for _ in range(nits):
        p = KP(x, y, p)

    torch.cuda.synchronize()
    end = time()
    print("Timing with {} points: {} x {:.4f}s".format(N, nits, (end - start) / nits))
