import math
import torch
from pykeops.torch import LazyTensor

M, N, D, DV = 1000, 1000, 3, 1

dtype = torch.float32

device_id = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

x = torch.rand(M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, requires_grad=True, device=device_id, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y) ** 2).sum(dim=2)
    Kxy = (-Dxy).exp()
    if backend == "keops":
        out = LazyTensor.__matmul__(Kxy, b, backend="CPU")
    else:
        out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    # print("out:",out.flatten()[:10])
    return out


backends = ["keops", "torch"]  # "keops_old"

out = []
for backend in backends:
    out.append(fun(x, y, b, backend).squeeze())

out_g = []
for k, backend in enumerate(backends):
    out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [b], create_graph=True)[0])

out_g2 = []
for k, backend in enumerate(backends):
    out_g2.append(torch.autograd.grad((out_g[k] ** 2).sum(), [b])[0])


class TestClass:
    def test_lazytensor_gaussian_cpu(self):
        assert torch.allclose(out[0], out[1])

    def test_lazytensor_gaussian_cpu_bw1(self):
        assert torch.allclose(out_g[0], out_g[1])

    def test_lazytensor_gaussian_cpu_bw2(self):
        assert torch.allclose(out_g2[0], out_g2[1])
