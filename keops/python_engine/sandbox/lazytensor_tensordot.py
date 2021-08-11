"""
=========
TensorDot
=========

This is a test script to showcase the tensordot syntax.
"""

import numpy as np
import torch

from pykeops.torch import LazyTensor


M, N = 2, 10

#######################################################################################################################
# Matrix multiplication as a special case of Tensordot
# ----------------------------------------------------
#
device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
do_warmup = True

a = torch.randn(4 * 7, requires_grad=True, device=device_id, dtype=torch.float64)
b = torch.randn(7, requires_grad=True, device=device_id, dtype=torch.float64)
c = a.reshape(4, 7) @ b

#######################################################################################################################
# A single matrix multiplication
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In this case no need to use KeOps: this is a sanity check.

A = LazyTensor(a[None, None, :])
B = LazyTensor(b[None, None, :])
C = A.keops_tensordot(B, (4, 7), (7,), (1,), (0,)).sum_reduction(dim=1)

# print(C, c)
print(
    "Compare the two MatVecMul implementations. All good?",
    torch.allclose(c.flatten(), C.flatten()),
)
