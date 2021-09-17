# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import LazyTensor

B1, B2, M, N, D, DV = 2, 3, 200, 300, 3, 300

dtype = torch.float32
sum_scheme = "block_sum"

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
do_warmup = False

x = torch.rand(B1, B2, M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(B1, 1, 1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(1, B2, N, DV, device=device_id, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y).square()).sum(dim=4)
    Kxy = (-Dxy).exp()
    if "keops" in backend:
        out = Kxy.__matmul__(b, sum_scheme=sum_scheme)
    else:
        out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    # print("out:",out[:,:10])
    return out


backends = ["keops", "torch"]

out = []
for backend in backends:
    if do_warmup:
        fun(
            x[:, :, : min(M, 100), :, :],
            y[:, :, :, : min(N, 100), :],
            b[:, :, : min(N, 100), :],
            backend,
        )
        fun(
            x[:, :, : min(M, 100), :, :],
            y[:, :, :, : min(N, 100), :],
            b[:, :, : min(N, 100), :],
            backend,
        )
    start = time.time()
    out.append(fun(x, y, b, backend).squeeze())
    end = time.time()
    print("time for " + backend + ":", end - start)

if len(out) > 1:
    print("relative error:", (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item())
