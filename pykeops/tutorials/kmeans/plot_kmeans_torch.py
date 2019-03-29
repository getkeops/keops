"""
================================
K-means clustering - PyTorch API
================================

The :func:`pykeops.torch.generic_argmin` routine allows us
to perform **bruteforce nearest neighbor search** with four lines of code.
It can thus be used to implement a **large-scale** 
`K-means clustering <https://en.wikipedia.org/wiki/K-means_clustering>`_,
**without memory overflows**.

.. note::
  For large and high dimensional datasets, this script 
  **outperforms its NumPy counterpart**
  as it avoids transfers between CPU (host) and GPU (device) memories.

  
"""

########################################################################
# Setup 
# -----------------
# Standard imports:

import time

import torch
from matplotlib import pyplot as plt

from pykeops.torch import Genred

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}
########################################################################
# Simple implementation of the K-means algorithm:


def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # Define our KeOps kernel:
    nn_search = Genred(
        'SqDist(x,y)',            # A simple squared L2 distance
        ['x = Vi({})'.format(D),  # target points of dimension D, indexed by "i"
         'y = Vj({})'.format(D)], # source points of dimension D, indexed by "j"
        reduction_op='ArgMin',
        axis=1,                   # The reduction is performed on the second axis
        cuda_type=dtype)          # "float32" and "float64" are available
    
    # K-means loop:
    # - x  is the point cloud, 
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization

    for i in range(Niter):
        cl  = nn_search(x,c).long().view(-1)  # Points -> Nearest cluster
        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print("K-means example with {} points in dimension {}, K = {}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format( 
                Niter, end - start, Niter, (end-start) / Niter))

    return cl, c
        

###############################################################
# K-means in 2D
# ----------------------
# First experiment with 10,000 points in dimension 2, with 50 classes:
#
N, D, K = 10000, 2, 50

###############################################################
# Define our dataset:
x = torch.randn(N, D, dtype=torchtype[dtype]) / 6 + .5

###############################################################
# Perform the computation:
cl, c = KMeans(x, K)

###############################################################
# Fancy display:

plt.figure(figsize=(8,8))
plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c=cl.cpu(), s= 30000 / len(x), cmap="tab10")
plt.scatter(c[:, 0].cpu(), c[:, 1].cpu(), c='black', s=50, alpha=.8)
plt.axis([0,1,0,1]) ; plt.tight_layout() ; plt.show()
 

####################################################################
# K-means in dimension 100
# -------------------------
# Second experiment with 1,000,000 points in dimension 100, with 1,000 classes:

if use_cuda:
    N, D, K = 1000000, 100, 1000
    x = torch.randn(N, D, dtype=torchtype[dtype])
    cl, c = KMeans(x, K)
