"""
=========
TensorDot
=========

This is a test script to showcase the tensordot syntax.
"""

import numpy as np
import torch

from pykeops.torch import LazyTensor

import pykeops.config

pykeops.config.gpu_available = False

M, N = 2, 10


#######################################################################################################################
# A Fourth example
# ^^^^^^^^^^^^^^^^

x = torch.randn(M, 20, 30, 4, 2, 20, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 20, 4, 5, 30, 20, requires_grad=True, dtype=torch.float64)

xshape, yshape = x.shape[1:], y.shape[1:]
f_keops = LazyTensor(x.reshape(M, 1, int(np.array((xshape)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(yshape).prod()))),
    xshape,
    yshape,
    (0, 1, 4),
    (0, 3, 4),
)
sum_f_keops = f_keops.sum_reduction(dim=1)
sum_f_torch2 = torch.tensordot(x, y, dims=([1, 2, 5], [1, 4, 5])).sum(3)
# sum_f_torch2 = torch.tensordot(x, y, dims=([3], [1])).sum(3)

print(sum_f_keops.flatten(), sum_f_torch2.flatten())
print(
    "Compare the two tensordot implementation. All good ????!",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()

print(
    "Compare the two gradient x tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, e, retain_graph=True)[0].numpy()
print(
    "Compare the two gradient y tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

print("------------------------------------------")
