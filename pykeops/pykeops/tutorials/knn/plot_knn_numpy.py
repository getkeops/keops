"""
=================================
K-NN classification - NumPy API
=================================

The :meth:`pykeops.numpy.LazyTensor.argKmin` reduction supported by KeOps :class:`pykeops.numpy.LazyTensor` allows us
to perform **bruteforce k-nearest neighbors search** with four lines of code.
It can thus be used to implement a **large-scale** 
`K-NN classifier <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_,
**without memory overflows**.



"""

#############################
# Setup
# -----------------
# Standard imports:

import time

import numpy as np
from matplotlib import pyplot as plt

from pykeops.numpy import LazyTensor
import pykeops.config

dtype = "float32"

#############################
# Dataset, in 2D:

N, D = (
    10000 if pykeops.config.gpu_available else 1000,
    2,
)  # Number of samples, dimension
x = np.random.rand(N, D).astype(dtype)  # Random samples on the unit square

# Random-ish class labels:
def fth(x):
    return 3 * x * (x - 0.5) * (x - 1) + x


cl = x[:, 1] + 0.1 * np.random.randn(N).astype(dtype) < fth(x[:, 0])

#############################
# Reference sampling grid, on the unit square:

M = 1000 if pykeops.config.gpu_available else 100
tmp = np.linspace(0, 1, M).astype(dtype)
g1, g2 = np.meshgrid(tmp, tmp)
g = np.hstack((g1.reshape(-1, 1), g2.reshape(-1, 1)))


#############################
# K-Nearest Neighbors search
# ----------------------------

##############################
# Peform the K-NN classification, with a fancy display:
#

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.scatter(x[:, 0], x[:, 1], c=cl, s=2)
plt.imshow(np.ones((2, 2)), extent=(0, 1, 0, 1), alpha=0)
plt.axis("off")
plt.axis([0, 1, 0, 1])
plt.title("{:,} data points,\n{:,} grid points".format(N, M * M))

for (i, K) in enumerate((1, 3, 10, 20, 50)):

    start = time.time()  # Benchmark:

    G_i = LazyTensor(g[:, None, :])  # (M**2, 1, 2)
    X_j = LazyTensor(x[None, :, :])  # (1, N, 2)
    D_ij = ((G_i - X_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
    indKNN = D_ij.argKmin(K, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor

    clg = np.mean(cl[indKNN], axis=1) > 0.5  # Classify the Grid points
    end = time.time()

    plt.subplot(2, 3, i + 2)  # Fancy display:
    clg = np.reshape(clg, (M, M))
    plt.imshow(clg, extent=(0, 1, 0, 1), origin="lower")
    plt.axis("off")
    plt.axis([0, 1, 0, 1])
    plt.tight_layout()
    plt.title("{}-NN classifier,\n t = {:.2f}s".format(K, end - start))

plt.show()
