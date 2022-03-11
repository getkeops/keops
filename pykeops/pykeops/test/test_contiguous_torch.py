from pykeops.torch import LazyTensor
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

a1 = torch.rand(2, 1000, 5)
a2 = ((a1.permute(2, 0, 1)).contiguous()).permute(1, 2, 0)

b = torch.rand(2, 1000, 5)
c = torch.rand(2, 1000, 5)

b_j = LazyTensor(b[:, None])

a1_i = LazyTensor(a1[:, :, None])
dist1 = a1_i.sqdist(b_j)
kernel1 = dist1.exp()
d1 = kernel1 @ c

a2_i = LazyTensor(a2[:, :, None])
dist2 = a2_i.sqdist(b_j)
kernel2 = dist2.exp()
d2 = kernel2 @ c


def test_contiguous_torch():
    assert torch.allclose(d2, d1)
