import torch
import pykeops.config as pykeopsconfig
from pykeops.torch import LazyTensor
from pykeops.test import assert_torch_allclose

M, N = 2, 10

# Matrix multiplication as a special case of Tensordot
torch.backends.cuda.matmul.allow_tf32 = False
use_cuda = torch.cuda.is_available() and pykeopsconfig.gpu_available
device_id = "cuda" if use_cuda else "cpu"

torch.manual_seed(0)
a = torch.randn(4 * 7, requires_grad=True, device=device_id, dtype=torch.float64)
b = torch.randn(7, requires_grad=True, device=device_id, dtype=torch.float64)
c = a.reshape(4, 7) @ b

A = LazyTensor(a[None, None, :])
B = LazyTensor(b[None, None, :])
C = A.keops_tensordot(B, (4, 7), (7,), (1,), (0,)).sum_reduction(dim=1)


def test_tensordot():
    assert_torch_allclose(c.flatten(), C.flatten(), label="lazytensor_tensordot")
