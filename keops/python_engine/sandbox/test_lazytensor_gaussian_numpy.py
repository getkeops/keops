# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
import numpy as np
from pykeops.numpy import LazyTensor

M, N, D, DV = 3000, 2000, 3, 1

dtype = np.float32

do_warmup = False

x = np.random.rand(M, 1, D).astype(dtype) / math.sqrt(D)
y = np.random.rand(1, N, D).astype(dtype) / math.sqrt(D)
b = np.random.randn(N, DV).astype(dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y)).sum(axis=2)
    if backend == "keops":
        Kxy = (-Dxy).exp()
    else:
        Kxy = np.exp(-Dxy)
    out = Kxy @ b
    # print("out:",out.flatten())
    return out


backends = ["keops", "numpy"]

out = []
for backend in backends:
    if do_warmup:
        fun(
            x[: min(M, 100), :, :], y[:, : min(N, 100), :], b[: min(N, 100), :], backend
        )
        fun(
            x[: min(M, 100), :, :], y[:, : min(N, 100), :], b[: min(N, 100), :], backend
        )
    start = time.time()
    out.append(fun(x, y, b, backend).squeeze())
    end = time.time()
    print("time for " + backend + ":", end - start)

if len(out) > 1:
    print("relative error:", (np.linalg.norm(out[0] - out[1]) / np.linalg.norm(out[0])))
