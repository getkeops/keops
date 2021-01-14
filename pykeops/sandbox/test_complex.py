
import time

import math
import torch
from pykeops.torch import LazyTensor

dtype = torch.complex64

M, N, D = 5, 5, 2

test_grad = True

device_id = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

do_warmup = False

x = torch.rand(M, 1, D, dtype=dtype, requires_grad=test_grad, device=device_id)
y = torch.rand(1, N, D, dtype=dtype, device=device_id)
a = -1.23
b = 1.54


def view_as_real(x):
    if torch.is_complex(x):
        return torch.view_as_real(x)
    else:
        return x


def fun(x, y, a, b, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
        conj = LazyTensor.conj
        angle = LazyTensor.angle
    elif backend == "torch":
        conj = torch.conj
        angle = torch.angle
    z = x-y
    Kxy = z.sum(dim=2).exp()*1j
    return Kxy.sum(dim=0)
    
backends = ["keops","torch"]

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
    #print(out[0])
    #print(out[1])
    print("relative error:", (torch.norm(view_as_real(out[0] - out[1]).cpu()) / torch.norm(view_as_real(out[0]).cpu())).item())

if test_grad:
    out_g = []
    for k, backend in enumerate(backends):
        start = time.time()
        if out[k].is_complex():
            out_g.append(torch.autograd.grad((out[k].real**2+out[k].imag**2).sum(), [x])[0])
        else:
            out_g.append(torch.autograd.grad((out[k]**2).sum(), [x])[0])
        end = time.time()
        print("time for " + backend + " (grad):", end - start)

    if len(out_g) > 1:
        #print(out_g[0])
        #print(out_g[1])
        print(
            "relative error grad:",
            (torch.norm(view_as_real(out_g[0] - out_g[1]).cpu()) / torch.norm(view_as_real(out_g[0]).cpu())).item())
        
