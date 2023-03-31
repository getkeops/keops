# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import LazyTensor

M, N, D, DV = 2000, 4000, 3, 1

dtype = torch.float32
sum_scheme = "block_sum"

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

x = torch.rand(3, M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(1, 1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(1, N, DV, device=device_id, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x * y).square()).sum(dim=3)
    Kxy = (-Dxy).exp()
    if "keops" in backend:
        out = Kxy.__matmul__(b, sum_scheme=sum_scheme)
    else:
        out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    # print("out:",out.flatten()[:10])
    return out


backends = ["keops", "torch"]

out = []
for backend in backends:
    start = time.time()
    out.append(fun(x, y, b, backend).squeeze())
    end = time.time()
    print("time for " + backend + ":", end - start)

if len(out) > 1:
    print("relative error:", (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item())
