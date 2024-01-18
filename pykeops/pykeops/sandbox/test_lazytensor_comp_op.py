# Test for Clamp operation using LazyTensors

import time

import math
import torch
from pykeops.torch import LazyTensor

M, N, D = 5, 5, 3

dtype = torch.float16

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

do_warmup = True

x = torch.randn(M, 1, D, device=device_id, dtype=dtype)
y = torch.randn(1, N, 1, device=device_id, dtype=dtype)


def fun(x, y, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    Kxy = ((x < y) != (y >= 1)).sum(dim=2)
    # Kxy = (x-y).clamp(x,x+y).sum(dim=2)
    # Kxy = (x - y).sign().sum(dim=2)
    return Kxy.sum(dim=1).to(dtype)


backends = ["torch", "keops"]

out = []
for backend in backends:
    if do_warmup:
        fun(x[: min(M, 100), :, :], y[:, : min(N, 100), :], backend)
        fun(x[: min(M, 100), :, :], y[:, : min(N, 100), :], backend)
    start = time.time()
    out.append(fun(x, y, backend).squeeze())
    end = time.time()
    # print(out[-1].squeeze()[:10])
    print("time for " + backend + ":", end - start)

if len(out) > 1:
    print("relative error:", (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item())
