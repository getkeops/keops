"""
================================
 Nearest Neighbors Descent (NND) approximate nearest neighbors search - PyTorch API
================================

The :class:`pykeops.torch.NND` class supported by KeOps allows us
to perform **approximate nearest neighbor search** with four lines of code.

Euclidean and Manhattan metrics are supported.

.. note::
  NNDescent is not fully optimised and we recommend the use of IVF-Flat instead.
  Nevertheless, we provide NNDescent as a means of benchmarking cutting edge nearest neighbor search algorithms

"""

########################################################################
# Setup
# -----------------
# Standard imports:

import time
import torch
from pykeops.torch import NND

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")
dtype = torch.float32 if use_cuda else torch.float64

########################################################################
# NNDescent search with Euclidean metric
# First experiment with N=$10^4$ points in dimension D=3 and 5 nearest neighbours, and default hyperparameters.

N, D, k = 10 ** 4, 3, 5

########################################################################
# Define our dataset:

torch.manual_seed(1)
x = 0.7 * torch.randn(N, D, dtype=dtype) + 0.3
y = 0.7 * torch.randn(N, D, dtype=dtype) + 0.3

########################################################################
# Create the NND class and fit the dataset:

nn = NNDescent(k=k)
nn.fit(x, queue=8)

########################################################################
# Query dataset search

approx_nn = nn.kneighbors(y)

########################################################################
# Define the function to compute the true nearest neighbors with brute force search


def brute_force(x, y, k, metric):
    x_i = LazyTensor(x.unsqueeze(0).to(device))
    y_j = LazyTensor(y.unsqueeze(1).to(device))
    if metric == "euclidean":
        D_ij = ((x_i - y_j) ** 2).sum(-1)
    elif metric == "manhattan":
        D_ij = ((x_i - y_j).abs()).sum(-1)
    indices = D_ij.argKmin(K=k, dim=1).long()
    return indices

   
########################################################################
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


########################################################################
# Compute the true nearest neighbors with brute force search using Euclidean distance

indices = brute_force(x=x, y=y, k=k, metric="euclidean")

########################################################################
# Check the performance of our algorithm

print("NND Recall:", accuracy(approx_nn.to(device), indices))

########################################################################
# Timing the algorithms to observe their performance

start = time.time()
iters = 10

# timing KeOps brute force
for _ in range(iters):
    indices = brute_force(x=x, y=y, k=k, metric="euclidean")
bf_time = time.time() - start
print(
    "KeOps brute force timing for", N, "points with", D, "dimensions:", bf_time / iters
)

# timing NNDescent
start = time.time()
for _ in range(iters):
    approx_nn = nn.kneighbors(y)
nnd_time = time.time() - start
print("KeOps NND timing for", N, "points with", D, "dimensions:", nnd_time / iters)

########################################################################
# NNDescent search using clusters and Manhattan distance
# Second experiment with N=$10^6$ points in dimension D=3, with 5 nearest neighbors and manhattan distance.

N, D, k = 10 ** 6, 3, 5

########################################################################
# Define our dataset:

torch.manual_seed(1)
x = 0.7 * torch.randn(N, D, dtype=dtype) + 0.3
x = x.to(device)
y = 0.7 * torch.randn(N, D, dtype=dtype) + 0.3
y = y.to(device)

########################################################################
# Create the NNDescent class and fit the dataset:

nn = NNDescent(k=k, metric="manhattan", initialization_method="cluster")
nn.fit(x, a=10, queue=5, clusters=64)

########################################################################
# Query dataset search

approx_nn = nn.kneighbors(y)

########################################################################
# Now computing the true nearest neighbors with brute force search using Manhattan distance

indices = brute_force(x=x, y=y, k=k, metric="manhattan")

########################################################################
# Check the performance of our algorithm

print("NND Recall:", accuracy(approx_nn.to(device), indices))

########################################################################
# Timing the algorithms to observe their performance

start = time.time()
iters = 10

# timing KeOps brute force
for _ in range(iters):
    indices = brute_force(x=x, y=y, k=k, metric="manhattan")
bf_time = time.time() - start
print(
    "KeOps brute force timing for", N, "points with", D, "dimensions:", bf_time / iters
)

# timing NNDescent
start = time.time()
for _ in range(iters):
    approx_nn = nn.kneighbors(y)
nnd_time = time.time() - start
print("KeOps NND timing for", N, "points with", D, "dimensions:", nnd_time / iters)
