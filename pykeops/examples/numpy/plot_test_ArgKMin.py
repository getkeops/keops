"""
Arg-K-Min reduction
===================

Using the :mod:`pykeop.numpy` API, we define a dataset of N points in :math:`\mathbb R^D` and compute for each
point the indices of its K nearest neighbours (including itself).
"""

###############################################################
# Setup
# ----------
#
# Standard imports:

import time
import numpy as np
from pykeops.numpy import Genred

###############################################################
# Define our dataset:

N = 10000  # Number of points
D = 2      # Dimension of the ambient space
K = 3      # Number of neighbors to look for

type = 'float32'  # May be 'float32' or 'float64'

x = np.random.randn(N,D).astype(type)

###############################################################
# KeOps Kernel
# -------------

formula   =  'SqDist(x,y)'          # Use a simple Euclidean (squared) norm
variables = ['x = Vx('+str(D)+')',  # First arg : i-variable, of size D
             'y = Vy('+str(D)+')']  # Second arg: j-variable, of size D

# N.B.: The number K is specified as an optional argument `opt_arg`
my_routine = Genred(formula, variables, reduction_op='ArgKMin', axis=1, 
                    cuda_type=type, opt_arg=K)


###############################################################
# Using our new :func:`pykeops.numpy.Genred` routine, 
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
my_routine( np.random.rand(10,D).astype(type),
            np.random.rand(10,D).astype(type) )

# Actually perform our K-nn search:
start = time.time()
c = my_routine(x, x, backend="auto").astype(int)
print("Time to perform the K-nn search: ",round(time.time()-start,5),"s")

# The result is now an (N,K) array of integers:
print("Output values :")
print(c)

