"""
=========================================================
IVF-Flat approximate nearest neighbors search - Numpy API
=========================================================

The :class:`pykeops.torch.IVF` class supported by KeOps allows us
to perform **approximate nearest neighbor search** with four lines of code.
It can thus be used to compute a **large-scale** nearest neighbors search **much faster**.
The code is based on the IVF-Flat algorithm and uses KeOps' block-sparse reductions to speed up the search by reducing the search space.

Euclidean, Manhattan and Angular metrics are supported.

.. note::
  Hyperbolic and custom metrics are not supported in the Numpy API, please use the PyTorch API instead.

"""

########################################################################
# Setup
# -----------------
# Standard imports:

import time
import numpy as np
from pykeops.numpy import IVF
from pykeops.numpy.utils import numpytools

########################################################################
# IVF nearest neighbour search with Euclidean metric
# First experiment with N=$10^5$ points in dimension D=3 and 5 nearest neighbours

N, D, k = 10 ** 5, 3, 5

########################################################################
# Define our dataset:

np.random.seed(1)
x = 0.7 * np.random.randn(N, D) + 0.3
y = 0.7 * np.random.randn(N, D) + 0.3

########################################################################
# Create the IVF class and fit the dataset:

nn = IVF(k=k)
# set the number of clusters in K-Means to 50
# set the number of nearest clusters we search over during the final query search to 5
nn.fit(x, clusters=50, a=5)

########################################################################
# Query dataset search

approx_nn = nn.kneighbors(y)

########################################################################
# Now computing the true nearest neighbors with brute force search

true_nn = nn.brute_force(x, y, k=k)

########################################################################
# Check the performance of our algorithm

print("IVF Recall:", numpytools.knn_accuracy(approx_nn, true_nn))

########################################################################
# Timing the algorithms to observe their performance

start = time.time()
iters = 10

# timing KeOps brute force
for _ in range(iters):
    true_nn = nn.brute_force(x, y, k=k)
bf_time = time.time() - start
print(
    "KeOps brute force timing for", N, "points with", D, "dimensions:", bf_time / iters
)

# timing IVF
nn = IVF(k=k)
nn.fit(x)
start = time.time()
for _ in range(iters):
    approx_nn = nn.kneighbors(y)
ivf_time = time.time() - start
print("KeOps IVF-Flat timing for", N, "points with", D, "dimensions:", ivf_time / iters)

########################################################################
# IVF nearest neighbors search with angular metric
# Second experiment with N=$10^5$ points in dimension D=3, with 5 nearest neighbors

np.random.seed(1)
x = 0.7 * np.random.randn(N, D) + 0.3
y = 0.7 * np.random.randn(N, D) + 0.3

# normalising the inputs to have norm of 1
x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
y_norm = y / np.linalg.norm(x, axis=1, keepdims=True)

nn = IVF(metric="angular")
true_nn = nn.brute_force(x_norm, y_norm)

nn = IVF(metric="angular")
nn.fit(x_norm)
approx_nn = nn.kneighbors(y_norm)
print("IVF Recall:", numpytools.knn_accuracy(approx_nn, true_nn))

########################################################################
# The IVF class also has an option to automatically normalise all inputs

nn = IVF(metric="angular", normalise=True)
nn.fit(x)
approx_nn = nn.kneighbors(y)
print("IVF Recall:", numpytools.knn_accuracy(approx_nn, true_nn))

########################################################################
# There is also an option to use full angular metric "angular_full", which uses the full angular metric. "angular" simply uses the dot product.

nn = IVF(metric="angular_full")
nn.fit(x)
approx_nn = nn.kneighbors(y)
print("IVF Recall:", numpytools.knn_accuracy(approx_nn, true_nn))
