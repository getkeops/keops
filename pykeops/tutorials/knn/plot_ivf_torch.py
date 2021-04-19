"""
===========================================================
IVF-Flat approximate nearest neighbors search - PyTorch API
===========================================================

The :class:`pykeops.torch.IVF` class supported by KeOps allows us
to perform **approximate nearest neighbor search** with four lines of code.
It can thus be used to compute a **large-scale** nearest neighbors search **much faster**.
The code is based on the IVF-Flat algorithm and uses KeOps' block-sparse reductions to speed up the search by reducing the search space.

Euclidean, Manhattan, Angular and Hyperbolic metrics are supported along with custom metrics.

.. note::
  Hyperbolic and custom metrics require the use of an approximation during the K-Means step.
  This is to obtain the centroid locations since a closed form expression might not be readily available
"""

###############################################################
# Setup
# -----------------
# Standard imports:

import time
import torch
from pykeops.torch import IVF

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")
dtype = torch.float32 if use_cuda else torch.float64

###############################################################
# IVF nearest neighbour search with Euclidean metric
# --------------------------------------------------
# First experiment with N=$10^6$ points in dimension D=3 and 5 nearest neighbours


N, D, k = 10 ** 6, 3, 5

###############################################################
# Define our dataset:

torch.manual_seed(1)
x = 0.7 * torch.randn(N, D, dtype=dtype, device=device) + 0.3
y = 0.7 * torch.randn(N, D, dtype=dtype, device=device) + 0.3

###############################################################
# Create the IVF class and fit the dataset:

nn = IVF(k=k)
# set the number of clusters in K-Means to 50
# set the number of nearest clusters we search over during the final query search to 5
nn.fit(x, clusters=50, a=5)

###############################################################
# Query dataset search

approx_nn = nn.kneighbors(y)

###############################################################
# Now computing the true nearest neighbors with brute force search

true_nn = nn.brute_force(x, y, k=k)

###############################################################
# Define the function to compute recall of the nearest neighbors


def accuracy(indices_test, indices_truth):
    """
    Compares the test and ground truth indices (rows = KNN for each point in dataset)
    Returns accuracy: proportion of correct nearest neighbours
    """
    N, k = indices_test.shape

    # Calculate number of correct nearest neighbours
    accuracy = 0
    for i in range(k):
        accuracy += torch.sum(indices_test == indices_truth).float() / N
        indices_truth = torch.roll(
            indices_truth, 1, -1
        )  # Create a rolling window (index positions may not match)
    accuracy = float(accuracy / k)  # percentage accuracy

    return accuracy


###############################################################
# Check the performance of our algorithm

print("IVF Recall:", accuracy(approx_nn, true_nn))

###############################################################
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

###############################################################
# IVF nearest neighbors search with angular metric
# Second experiment with N=$10^6$ points in dimension D=3, with 5 nearest neighbors

torch.manual_seed(1)
x = 0.7 * torch.randn(N, D, dtype=dtype, device=device) + 0.3
y = 0.7 * torch.randn(N, D, dtype=dtype, device=device) + 0.3

# normalising the inputs to have norm of 1
x_norm = x / torch.linalg.norm(x, dim=1, keepdim=True)
y_norm = y / torch.linalg.norm(y, dim=1, keepdim=True)

nn = IVF(metric="angular")
true_nn = nn.brute_force(x_norm, y_norm)

nn = IVF(metric="angular")
nn.fit(x_norm)
approx_nn = nn.kneighbors(y_norm)
print("IVF Recall:", accuracy(approx_nn, true_nn))

###############################################################
# The IVF class also has an option to automatically normalise all inputs

nn = IVF(metric="angular", normalise=True)
nn.fit(x)
approx_nn = nn.kneighbors(y)
print("IVF Recall:", accuracy(approx_nn, true_nn))

###############################################################
# There is also an option to use full angular metric "angular_full", which uses the full angular metric. "angular" simply uses the dot product.

nn = IVF(metric="angular_full")
nn.fit(x)
approx_nn = nn.kneighbors(y)
print("IVF Recall:", accuracy(approx_nn, true_nn))

###############################################################
# IVF nearest neighbors search with approximations for K-Means centroids
# We run two experiment with N=$10^6$ points in dimension D=3, with 5 nearest neighbors. The first uses the hyperbolic metric while the second uses a custom metric.

# hyperbolic data generation
torch.manual_seed(1)
x = 0.5 + torch.rand(N, D, dtype=dtype, device=device)
y = 0.5 + torch.rand(N, D, dtype=dtype, device=device)

nn = IVF(metric="hyperbolic")
# set approx to True
# n is the number of times we run gradient descent steps for the approximation, default of 50
nn.fit(x, approx=True, n=50)
approx_nn = nn.kneighbors(y)
true_nn = nn.brute_force(x, y)
print("IVF Recall:", accuracy(approx_nn, true_nn))

# define a custom metric
def minkowski(x, y, p=3):
    """Returns the computation of a metric
    Note the shape of the input tensors the function should accept

    Args:
      x (tensor): Input dataset of size 1, N, D
      y (tensor): Query dataset of size M, 1, D

    """
    return ((x - y).abs() ** p).sum(-1)


# testing custom metric
nn = IVF(metric=minkowski)
nn.fit(x, approx=True)
approx_nn = nn.kneighbors(y)
true_nn = nn.brute_force(x, y)
print("IVF Recall:", accuracy(approx_nn, true_nn))
