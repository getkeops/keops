# Test for Clamp operation using LazyTensors

import time

import math
import torch
from pykeops.torch import LazyTensor

dtype = torch.float16

M, N, D = 1001, 1001, 3

test_grad = False

device_id = "cuda" if torch.cuda.is_available() else "cpu"

do_warmup = False

x = torch.randn(M, 1, D, dtype=dtype, requires_grad=test_grad, device=device_id)
y = torch.randn(1, N, D, dtype=dtype, device=device_id)
a = -1.23
b = 1.54


def fun(x, y, a, b, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    elif backend != "torch":
        raise ValueError("wrong backend")
    Dxy = ((x * y)).sum(dim=2)
    Kxy = (-(Dxy ** 2)).exp()
    return Kxy.sum(dim=0)


backends = ["torch", "keops"]

out = []
for backend in backends:
    if do_warmup:
        fun(x[: min(M, 100), :, :], y[:, : min(N, 100), :], a, b, backend)
        fun(x[: min(M, 100), :, :], y[:, : min(N, 100), :], a, b, backend)
    start = time.time()
    out.append(fun(x, y, a, b, backend).squeeze())
    end = time.time()
    print("time for " + backend + ":", end - start)

if len(out) > 1:
    print("relative error:", (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item())

if test_grad:
    out_g = []
    for k, backend in enumerate(backends):
        start = time.time()
        out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [x])[0])
        end = time.time()
        print("time for " + backend + " (grad):", end - start)

    if len(out_g) > 1:
        print(
            "relative error grad:",
            (torch.norm(out_g[0] - out_g[1]) / torch.norm(out_g[0])).item(),
        )
