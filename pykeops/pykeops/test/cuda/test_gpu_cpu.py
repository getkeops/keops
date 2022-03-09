import math
import torch
from pykeops.torch import LazyTensor

M, N, D, DV = 1000, 1000, 3, 1

dtype = torch.float32
device_id = "cpu"

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


out = []
for backend in ["torch", "keops_cpu", "keops_gpu"]:
    out.append(fun(x, y, b, backend).squeeze())


def test_torch_keops_cpu():
    assert torch.allclose(out[0], out[1])

def test_torch_keops_gpu():
    assert torch.allclose(out[0], out[2])