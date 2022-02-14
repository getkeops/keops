# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import LazyTensor

M, N, D, DV = 1000, 1000, 3, 1

dtype = torch.float32

device_id = "cpu"  # "cuda:1" if torch.cuda.is_available() else "cpu"
do_warmup = True

x = torch.rand(M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y) ** 2).sum(dim=2)
    Kxy = (-Dxy).exp()
    if "keops" in backend:
        if backend.split("_")[1] == "gpu":
            out = Kxy.__matmul__(b, backend="GPU_1D")
        elif backend.split("_")[1] == "cpu":
            out = Kxy.__matmul__(b, backend="CPU")
    else:
        out = Kxy @ b
    return out


backends = ["torch", "keops_cpu", "keops_gpu"]

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

for k in range(1, len(out)):
    print(
        f"relative error for {backends[k]}:",
        (torch.norm(out[0] - out[k]) / torch.norm(out[0])).item(),
    )
