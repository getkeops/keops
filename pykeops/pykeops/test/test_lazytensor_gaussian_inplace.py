import math
import torch
import pykeops.config
from pykeops.torch import LazyTensor
from pykeops.test import assert_torch_allclose

M, N, D, DV = 20, 30, 3, 1

dtype = torch.float32
sum_scheme = "block_sum"

torch.backends.cuda.matmul.allow_tf32 = False
use_cuda = torch.cuda.is_available() and pykeops.config.cuda.is_available()
device_id = "cuda" if use_cuda else "cpu"

torch.manual_seed(0)
x = torch.rand(M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)
a = torch.empty(M, DV, device=device_id, dtype=dtype)


def fun(x, y, b, backend, out=None):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y).square()).sum(dim=2)
    Kxy = (-Dxy).exp()
    if "keops" in backend:
        Kxy.__matmul__(b, sum_scheme=sum_scheme, out=out)
    else:
        out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    # print("out:",out)
    return out


out = []
for backend in ["keops", "torch"]:
    out.append(fun(x, y, b, backend, out=a).squeeze())


def test_lazytensor_gaussian_inplace():
    assert_torch_allclose(out[0], out[1], label="gaussian_inplace")
