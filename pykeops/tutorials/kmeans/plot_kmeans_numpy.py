"""
===============================
K-means clustering - NumPy API
===============================

We define a dataset of :math:`N` points in :math:`\mathbb R^D`, then apply a simple k-means algorithm.
This example uses a pure NumPy framework (without Pytorch).
"""

#############################
# Setup 
# -----------------
# Standard imports:

import time
import numpy as np
from pykeops.numpy import generic_argmin
from pykeops.numpy.utils import IsGpuAvailable

from matplotlib import pyplot as plt
type = 'float32'  # May be 'float32' or 'float64'

#######################################
# Simple implementation of the K-means algorithm:

def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # Define our KeOps kernel:
    my_routine = generic_argmin( 
        'SqDist(x,y)',  # A simple squared L2 distance
        'ind = Vx(1)',  # The output index is indexed by "i"
        'x = Vx({})'.format(D),  # 1st arg: target points of dimension D, indexed by "i"
        'y = Vy({})'.format(D),  # 2nd arg: source points of dimension D, indexed by "j"
        cuda_type = type )  # "float32" and "float64" are available
    
    # Dummy first call for accurate timing (GPU warmup):
    dum = np.random.rand(10,D).astype(type)
    my_routine(dum,dum)
    
    # K-means loop:
    # - x  is the point cloud, 
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = np.copy(x[:K, :])  # Simplistic random initialization

    for i in range(Niter):
        cl = my_routine(x,c).astype(int).reshape(N)  # Points -> Nearest cluster
        Ncl = np.bincount(cl).astype(type)           # Weight of each class   
        for d in range(D):  # Recompute the cluster centroids with np.bincount...
            c[:, d] = np.bincount(cl, weights=x[:, d]) / Ncl

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

#####################
# Define our dataset:
x = np.random.randn(N, D).astype(type) / 6 + .5

#####################
# Perform the computation:
cl, c = KMeans(x, K)

#####################
# Fancy display:

plt.figure(figsize=(8,8))
plt.scatter(x[:, 0], x[:, 1], c=cl, s= 30000 / len(x), cmap="tab10")
plt.scatter(c[:, 0], c[:, 1], c='black', s=50, alpha=.8)
plt.axis([0,1,0,1]) ; plt.tight_layout() ; plt.show()
 

####################################################################
# K-means in dimension 100
# -------------------------
# Second experiment with 1,000,000 points in dimension 100, with 1,000 classes:

if IsGpuAvailable():
    N, D, K = 1000000, 100, 1000
    x = np.random.randn(N, D).astype(type)
    cl, c = KMeans(x, K)
