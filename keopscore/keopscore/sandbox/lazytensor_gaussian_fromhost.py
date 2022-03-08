# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import LazyTensor

M, N, D, DV = 2500, 2000, 3, 1
# M, N, D, DV = 2, 3, 3, 1

dtype = torch.float64
sum_scheme = "block_sum"

device_id = 0 if torch.cuda.is_available() else -1
do_warmup = True

x = torch.rand(M, 1, D, dtype=dtype) / math.sqrt(D)
y = torch.rand(1, N, D, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y).square()).sum(dim=2)
    Kxy = (-Dxy).exp()
    if "keops" in backend:
        out = Kxy.__matmul__(b, sum_scheme=sum_scheme, device_id=device_id)
    else:
        out = Kxy @ b
    # print("out:",out)
    return out


backends = ["keops", "torch"]

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
    print("relative error:", (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item())
