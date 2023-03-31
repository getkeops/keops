import numpy as np
import torch
from pykeops.torch import ComplexLazyTensor

M, N = 2, 10

x = torch.randn(M, 2, 3, 2, 2, 4, dtype=torch.complex64)
y = torch.randn(N, 2, 4, 2, 3, 2, 3, dtype=torch.complex64)
xshape, yshape = x.shape[1:], y.shape[1:]
A = ComplexLazyTensor(x.reshape(M, 1, int(np.array((xshape)).prod())))
B = ComplexLazyTensor(y.reshape(1, N, int(np.array(yshape).prod())))
f_keops = A.keops_tensordot(
    B,
    xshape,
    yshape,
    (4, 0, 2),
    (1, 4, 2),
)
f_keops.sum_reduction(dim=1)
