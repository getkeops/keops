# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import LazyTensor

M, N, D, DV = 10000, 10000, 3, 10000

dtype = torch.float32

test_grad = True
test_grad2 = False
device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
do_warmup = True

x = torch.rand(
    M, 1, D, requires_grad=test_grad, device=device_id, dtype=dtype
) / math.sqrt(D)
y = torch.rand(1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)


def fun(x, y, b, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    if backend == "keops_alt":
        x = LazyTensor(x[None, ...])  # (M,1,D) -> (1,M,1,D)
        y = LazyTensor(y[None, ...])  # (1,N,D) -> (1,1,N,D)
    if backend == "keops_alt":
        Dxy = ((x - y) ** 2).sum(dim=3)
    else:
        Dxy = ((x - y) ** 2).sum(dim=2)
    Kxy = (-Dxy).exp()
    if backend == "torch":
        out = Kxy @ b
    elif backend == "keops":
        b = LazyTensor(b[None, ...])  # (N,DV) -> (1,N,DV)
        out = (Kxy * b).sum(dim=1)
    else:
        b = LazyTensor(
            b.transpose(0, 1).contiguous()[:, None, :, None]
        )  # (N,DV) -> (DV,1,N,1)
        out = (Kxy * b).sum(dim=2)  # (DV,M,1)
        out = out.transpose(0, 1)  # (DV,M,1) -> (M,DV,1)
        out = out.view(out.shape[:-1])  # (M,DV,1) -> (M,DV)
    if device_id != "cpu":
        torch.cuda.synchronize()
    return out


backends = ["torch", "keops", "keops_alt"]

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
    for k in range(1, len(out)):
        print(
            f"relative error:({backends[k]})",
            (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item(),
        )


if test_grad:
    out_g = []
    for k, backend in enumerate(backends):
        start = time.time()
        out_g.append(
            torch.autograd.grad((out[k] ** 2).sum(), [x], create_graph=True)[0]
        )
        end = time.time()
        print("time for " + backend + " (grad):", end - start)

    for k in range(1, len(out_g)):
        print(
            f"relative error grad:({backends[k]})",
            (torch.norm(out_g[0] - out_g[1]) / torch.norm(out_g[0])).item(),
        )

if test_grad2:
    out_g2 = []
    for k, backend in enumerate(backends):
        start = time.time()
        out_g2.append(torch.autograd.grad((out_g[k] ** 2).sum(), [x])[0])
        end = time.time()
        print("time for " + backend + " (grad):", end - start)

    for k in range(1, len(out_g2)):
        print(
            f"relative error grad2:({backends[k]})",
            (torch.norm(out_g2[0] - out_g2[1]) / torch.norm(out_g2[0])).item(),
        )
