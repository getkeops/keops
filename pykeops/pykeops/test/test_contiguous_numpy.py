from pykeops.numpy import LazyTensor
import numpy as np

np.random.seed(0)
a1 = np.random.rand(2, 1000, 5)
a2 = np.ascontiguousarray(a1.transpose(2, 0, 1)).transpose(1, 2, 0)

b = np.random.rand(2, 1000, 5)
c = np.random.rand(2, 1000, 5)

b_j = LazyTensor(b[:, None])

a1_i = LazyTensor(a1[:, :, None])
dist1 = a1_i.sqdist(b_j)
kernel1 = dist1.exp()
d1 = kernel1 @ c

a2_i = LazyTensor(a2[:, :, None])
dist2 = a2_i.sqdist(b_j)
kernel2 = dist2.exp()
d2 = kernel2 @ c


def test_contiguous_numpy():
    assert np.allclose(d2, d1)
