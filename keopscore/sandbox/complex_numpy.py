import time

import math
import numpy as np
from pykeops.numpy import LazyTensor, ComplexLazyTensor

M, N, D = 1000, 1000, 3

dtype = "float32"

do_warmup = False

x = np.random.rand(M, 1, D).astype(dtype) + 1j * np.random.rand(M, 1, D).astype(dtype)
y = np.random.rand(1, N, D).astype(dtype) + 1j * np.random.rand(1, N, D).astype(dtype)
a = -1.23
b = 1.54


def view_as_real(x):
    if x.dtype == complex:
        return torch.view_as_real(x)
    else:
        return x


def fun(x, y, a, b, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
        conj = ComplexLazyTensor.conj
        angle = ComplexLazyTensor.angle
    else:
        conj = np.conj
        angle = np.angle
    Kxy = ((x * y) * y.real + x + x.real).sum(axis=2)
    return Kxy.sum(axis=0)


backends = ["numpy", "keops"]

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
    # print(out[0])
    # print(out[1])
    print(
        "relative error:",
        (
            np.linalg.norm((out[0] - out[1]).view("float"))
            / np.linalg.norm((out[0]).view("float"))
        ).item(),
    )
