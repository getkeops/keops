import math

import pytest
import torch

from pykeops.torch import LazyTensor
from pykeops.test import assert_torch_allclose

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.rand(5, 1, dtype=torch.float64) * 2 * math.pi
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
    s2.backward()


@pytest.mark.skipif(torch.__version__ < "1.8", reason="Requires torch>=1.8")
def test_sinc():
    assert_torch_allclose(s1, s2, label="sinc")


@pytest.mark.skipif(torch.__version__ < "1.8", reason="Requires torch>=1.8")
def test_sinc_grad():
    assert_torch_allclose(x.grad, y.grad, label="sinc_grad")
