# Non-Uniform Discrete Fourier Tranform example

import time

import math
import torch
from pykeops.torch import LazyTensor

dtype = torch.float32
dtype_c = torch.complex64

M, N, D = 1000, 1000, 1

test_grad = False

device_id = "cuda" if torch.cuda.is_available() else "cpu"

do_warmup = False

x = torch.rand(1, N, D, dtype=dtype_c, requires_grad=test_grad, device=device_id)
p = torch.rand(1, N, D, dtype=dtype, device=device_id)
f = torch.rand(M, 1, D, dtype=dtype, device=device_id)


def view_as_real(x):
    if torch.is_complex(x):
        return torch.view_as_real(x)
    else:
        return x


def fun(x, p, f, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        p = LazyTensor(p)
        f = LazyTensor(f)
    X = x * (-2 * math.pi * 1j * p * f).exp()
    return X.sum(dim=0)


backends = ["keops", "torch"]

out = []
for backend in backends:
    if do_warmup:
        fun(
            x[:, : min(N, 100), :],
            p[:, : min(N, 100), :],
            f[: min(M, 100), :, :],
            backend,
        )
        fun(
            x[:, : min(N, 100), :],
            p[:, : min(N, 100), :],
            f[: min(M, 100), :, :],
            backend,
        )
    start = time.time()
    out.append(fun(x, p, f, backend).squeeze())
    end = time.time()
    print("time for " + backend + ":", end - start)

if len(out) > 1:
    print(
        "relative error:",
        (
            torch.norm(view_as_real(out[0] - out[1]).cpu())
            / torch.norm(view_as_real(out[0]).cpu())
        ).item(),
    )

if test_grad:
    out_g = []
    for k, backend in enumerate(backends):
        start = time.time()
        if out[k].is_complex():
            out_g.append(
                torch.autograd.grad((out[k].real ** 2 + out[k].imag ** 2).sum(), [x])[0]
            )
        else:
            out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [x])[0])
        end = time.time()
        print("time for " + backend + " (grad):", end - start)

    if len(out_g) > 1:
        print(
            "relative error grad:",
            (
                torch.norm(view_as_real(out_g[0] - out_g[1]).cpu())
                / torch.norm(view_as_real(out_g[0]).cpu())
            ).item(),
        )
