import torch
from pykeops.torch import LazyTensor

M, N, D = 1000, 1000, 3

torch.backends.cuda.matmul.allow_tf32 = False
device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
x = torch.randn(M, 1, D, requires_grad=True, device=device_id)
y = torch.randn(1, N, D, device=device_id)
a = -1.23
b = 1.54


def fun(x, y, a, b, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    elif backend != "torch":
        raise ValueError("wrong backend")
    Dxy = ((x * y).clamp(a, b)).sum(dim=2)
    Kxy = (-(Dxy**2)).exp()
    return Kxy.sum(dim=1)


out = []
for backend in ["torch", "keops"]:
    out.append(fun(x, y, a, b, backend).squeeze())

out_g = []
for k, backend in enumerate(["torch", "keops"]):
    out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [x])[0])


class TestCase:
    def test_lazytensor_clamp_fw(self):
        assert torch.allclose(out[0], out[1])

    def test_lazytensor_clamp_bw(self):
        assert torch.allclose(out_g[0], out_g[1], atol=0.01)
