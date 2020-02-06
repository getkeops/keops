"""
Arg-K-Min reduction
===================

Using the :mod:`pykeops.numpy` API, we define a dataset of N points in :math:`\mathbb R^D` and compute for each
point the indices of its K nearest neighbours (including itself).

 
"""

###############################################################
# Setup
# ----------
#
# Standard imports:

import time

import matplotlib.pyplot as plt
import numpy as np

from pykeops.numpy import Genred

###############################################################
# Define our dataset:

M, N = 10, 2000000  # Number of points
D = 2     # Dimension of the ambient space
K = 1     # Number of neighbors to look for

dtype = 'float32'  # May be 'float32' or 'float64'

x = np.random.rand(M,D).astype(dtype)
y = np.random.rand(N,D).astype(dtype)

###############################################################
# KeOps Kernel
# -------------

formula   =  'SqDist(x,y)'          # Use a simple Euclidean (squared) norm
variables = ['x = Vi('+str(D)+')',  # First arg : i-variable, of size D
             'y = Vj('+str(D)+')']  # Second arg: j-variable, of size D

# N.B.: The number K is specified as an optional argument `opt_arg`
my_routine = Genred(formula, variables, reduction_op='ArgMin', axis=1, 
                    dtype=dtype)


###############################################################
# Using our new :class:`pykeops.numpy.Genred` routine, 
# we perform a K-nearest neighbor search ( **reduction_op** = ``"ArgKMin"`` )
# over the :math:`j` variable :math:`y_j` ( **axis** = 1):
#
# .. note:: 
#   If CUDA is available and **backend** is ``"auto"`` or not specified,
#   KeOps will:
#
#     1. Load the data on the GPU
#     2. Perform the computation on the device 
#     3. Unload the result back to the CPU
#
#   as it is assumed to be most efficient for large-scale problems.
#   By specifying **backend** = ``"CPU"`` in the call to ``my_routine``, 
#   you can bypass this procedure and use a simple C++ ``for`` loop instead.

# Dummy first call to warm-up the GPU and thus get an accurate timing:
my_routine( np.random.rand(10,D).astype(dtype),
            np.random.rand(10,D).astype(dtype) )

# Actually perform our K-nn search:
start = time.time()
ind = my_routine(x, y, backend="auto").flatten()
print("Time to perform the K-nn search: ",round(time.time()-start,5),"s")


# compare with numpy implementation
start = time.time()
D2 = 0
for k in range(D):
    D2 += (x[:,k][:,None]-y[:,k][:,None].T)**2
ind2 = np.argmin(D2,axis=1)
print("Time to perform the K-nn search (numpy): ",round(time.time()-start,5),"s")

print("error : ",(ind-ind2).sum())

print(ind)

print(ind2)

print(ind-ind2)


'''
plt.figure(figsize=(8,8))
plt.scatter(x[:,0], x[:,1], s= 25*500 / len(x))
plt.scatter(y[:,0], y[:,1], s= 25*500 / len(y))

for k in range(K):  # Highlight some points and their nearest neighbors
    plt.scatter(y[ ind[:4,k], 0],y[ ind[:4,k], 1], s= 100)
    

plt.axis("equal") ; plt.axis([0,1,0,1])
plt.tight_layout(); plt.show()
'''