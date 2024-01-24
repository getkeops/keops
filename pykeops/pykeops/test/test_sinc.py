import pytest

import torch
import math
from pykeops.torch import LazyTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

x = torch.rand(5, 1) * 2 * math.pi
y = x.data.clone()
x = x.to(device)
y = y.to(device)
x.requires_grad = True
y.requires_grad = True

x_i = LazyTensor(x[:, None])
s1 = x_i.sinc().sum(0)
s1.backward()

if torch.__version__ >= "1.8":
    s2 = torch.sum(torch.sinc(y))

    assert torch.abs(s1 - s2) < 1e-3
    s2.backward()
    print("grad_s1 - grad_s2", torch.max(torch.abs(x.grad - y.grad)).item())
    assert torch.max(torch.abs(x.grad - y.grad)) < 1e-3


@pytest.mark.skipif(torch.__version__ < "1.8", reason="Requires torch>=1.8")
def test_sinc():
    assert torch.allclose(s1, s2), torch.abs(s1 - s2)


@pytest.mark.skipif(torch.__version__ < "1.8", reason="Requires torch>=1.8")
def test_sinc_grad():
    assert torch.allclose(x.grad, y.grad)
