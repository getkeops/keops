# Test for Clamp operation using LazyTensors

import torch
from pykeops.torch import LazyTensor

dtype = torch.float16

M, N, D = 5, 5, 1

device_id = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(M, 1, D, dtype=dtype, requires_grad=True, device=device_id)
y = torch.randn(1, N, D, dtype=dtype, device=device_id)


def fun(x, y, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    elif backend != "torch":
        raise ValueError("wrong backend")
    Dxy = (x - y).sum(dim=2)
    Kxy = Dxy
    return Kxy.sum(dim=0)


out = []
for backend in ["torch", "keops"]:
    out.append(fun(x, y, backend).squeeze())

out_g = []
for k, backend in enumerate(["torch", "keops"]):
    out_g.append(torch.autograd.grad(out[k][0], [x])[0])


def test_float16_fw():
    assert torch.allclose(out[0], out[1], atol=.001, rtol=.001)


def test_float16_bw():
    assert torch.allclose(out_g[0], out_g[1])
