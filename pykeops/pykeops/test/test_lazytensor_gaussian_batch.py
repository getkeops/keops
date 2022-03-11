import math
import torch
from pykeops.torch import LazyTensor

B1, B2, M, N, D, DV = 3, 4, 20, 25, 3, 2

torch.backends.cuda.matmul.allow_tf32 = False
device_id = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1)
x = torch.rand(1, B2, M, 1, D, device=device_id) / math.sqrt(D)
y = torch.rand(B1, B2, 1, N, D, device=device_id) / math.sqrt(D)
b = torch.randn(B1, 1, N, DV, requires_grad=True, device=device_id)


def fun(x, y, b, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    elif backend != "torch":
        raise ValueError("wrong backend")
    Dxy = ((x - y) ** 2).sum(dim=4)
    Kxy = (-Dxy).exp()
    out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    return out


backends = ["torch", "keops"]

out = []
for backend in backends:
    out.append(fun(x, y, b, backend).squeeze())

out_g = []
for k, backend in enumerate(backends):
    out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [b])[0])


class TestCase:
    def test_lazytensor_gaussian_batch_fw(self):
        # print(out[0]- out[1])
        assert torch.allclose(out[0], out[1], atol=1e-6)

    def test_lazytensor_gaussian_batch_bw(self):
        assert torch.allclose(out_g[0], out_g[1])
