import math
import torch
from pykeops.torch import LazyTensor

B1, B2, M, N, D, DV = 2, 3, 200, 300, 300, 1

dtype = torch.float32
sum_scheme = "block_sum"

torch.backends.cuda.matmul.allow_tf32 = False
device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
x = torch.rand(B1, B2, M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(B1, 1, 1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(1, B2, N, DV, device=device_id, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y).square()).sum(dim=4)
    Kxy = (-Dxy).exp()
    if "keops" in backend:
        out = Kxy.__matmul__(b, sum_scheme=sum_scheme)
    else:
        out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    # print("out:",out)
    return out


out = []
for backend in ["keops", "torch"]:
    out.append(fun(x, y, b, backend).squeeze())


def test_chunk_ranges():
    assert torch.allclose(out[0], out[1], rtol=0.0001, atol=0.0001)
