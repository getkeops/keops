"""
SumSoftMaxWeight reduction
==========================
"""

###############################################################################
# Using the :class:`numpy.Genred <pykeops.numpy.Genred>` class,
# we show how to perform a computation specified through:
#
# * Its **inputs**:
#
#     - :math:`x`, an array of size :math:`M\times 3` made up of :math:`M` vectors in :math:`\mathbb R^3`,
#     - :math:`y`, an array of size :math:`N\times 3` made up of :math:`N` vectors in :math:`\mathbb R^3`,
#     - :math:`b`, an array of size :math:`N\times 2` made up of :math:`N` vectors in :math:`\mathbb R^2`.
#
# * Its **output**:
#
#     - :math:`c`, an array of size :math:`M\times 2` made up of
#       :math:`M` vectors in :math:`\mathbb R^2` such that
#
#       .. math::
#
#           c_i = \frac{\sum_j \exp(K(x_i,y_j))\,\cdot\,b_j }{\sum_j \exp(K(x_i,y_j))},
#
#       with :math:`K(x_i,y_j) = \|x_i-y_j\|^2`.
#

###############################################################################
# Setup
# ----------------
#
# Standard imports:

import time
import numpy as np
from pykeops.numpy import Genred
import matplotlib.pyplot as plt

###############################################################################
# Define our dataset:
#

M = 5000  # Number of "i" points
N = 4000  # Number of "j" points
D = 3  # Dimension of the ambient space
Dv = 2  # Dimension of the vectors

x = 2 * np.random.randn(M, D)
y = 2 * np.random.randn(N, D)
b = np.random.rand(N, Dv)

###############################################################################
# KeOps kernel
# ---------------
#
# Create a new generic routine using the :class:`numpy.Genred <pykeops.numpy.Genred>`
# constructor:

formula = "SqDist(x,y)"
formula_weights = "b"
aliases = [
    "x = Vi(" + str(D) + ")",  # First arg:  i-variable of size D
    "y = Vj(" + str(D) + ")",  # Second arg: j-variable of size D
    "b = Vj(" + str(Dv) + ")",
]  # Third arg:  j-variable of size Dv

softmax_op = Genred(
    formula, aliases, reduction_op="SumSoftMaxWeight", axis=1, formula2=formula_weights
)

# Dummy first call to warmup the GPU and get accurate timings:
_ = softmax_op(x, y, b)

###############################################################################
# Use our new function on arbitrary Numpy arrays:
#

start = time.time()
c = softmax_op(x, y, b, backend="GPU_1D")
print("Timing (KeOps implementation): ", round(time.time() - start, 5), "s")

# compare with direct implementation
start = time.time()
cc = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
cc -= np.max(cc, axis=1)[:, None]  # Subtract the max to prevent numeric overflows
cc = np.exp(cc) @ b / np.sum(np.exp(cc), axis=1)[:, None]
print("Timing (Numpy implementation): ", round(time.time() - start, 5), "s")

print("Relative error : ", (np.linalg.norm(c - cc) / np.linalg.norm(c)).item())

# Plot the results next to each other:
for i in range(Dv):
    plt.subplot(Dv, 1, i + 1)
    plt.plot(c[:40, i], "-", label="KeOps")
    plt.plot(cc[:40, i], "--", label="NumPy")
    plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
