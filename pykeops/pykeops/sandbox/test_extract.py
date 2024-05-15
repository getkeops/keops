# Test for Clamp operation using LazyTensors

import time

import math
import torch
from pykeops.torch import LazyTensor

M, N, D = 1000, 1000, 10

test_grad = True

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

do_warmup = False

x = torch.randn(M, 1, D, requires_grad=test_grad, device=device_id)
y = torch.randn(1, N, D, device=device_id)


def fun(x, y, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
        Kxy = (-((x-y)**2).sum(axis=2)*x**2).exp()[:,:,2]
        return Kxy.sum(dim=1, sum_scheme="direct_acc")
    elif backend == "torch":
        Kxy = (-((x-y)**2).sum(axis=2, keepdim=True)*x**2).exp()[:,:,2]
        return Kxy.sum(dim=1)
    else:
        raise ValueError("wrong backend")

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
