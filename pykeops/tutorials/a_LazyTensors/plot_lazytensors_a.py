"""
================================================
A wrapper for NumPy and PyTorch arrays
================================================

KeOps is all about bringing **semi-symbolic** calculus
to modern computing libraries,
alleviating the need for **huge intermediate variables**
such as *kernel* or *distance* matrices in machine
learning and computational geometry.

"""

#########################################################################
# First steps 
# -----------------
#
# A simple high-level interface to the KeOps inner routines is provided by
# the :mod:`LazyTensor <pykeops.common.lazy_tensor.LazyTensor>`
# **symbolic wrapper**, which can be used in conjunction with
# both **NumPy arrays** or **PyTorch tensors**
# - just don't mix the two frameworks in the same computation!
#
# To illustrate its main features on a **simple**
# **example**, let's generate two point clouds :math:`(x_i)_{i\in[1,M]}` 
# and :math:`(y_j)_{j\in[1,N]}` in the unit square:

import numpy as np

M, N = 1000, 2000
x = np.random.rand(M, 2)
y = np.random.rand(N, 2)

##########################################################################
# With NumPy, an efficient way of computing the **nearest y-neighbor**
#
# .. math::
#   y_{\sigma(i)} ~=~ \arg \min_{j\in [1,N]} \| x_i - y_j\|^2
#
# for all points :math:`x_i` is to perform a :func:`numpy.argmin()`
# reduction on the **M-by-N matrix** of squared distances
#
# .. math::
#   D_{i,j} ~=~ \|x_i-y_j\|^2,
#
# computed using **tensorized**, broadcasted operators:
#

x_i = x[:,None,:]  # (M, 1, 2) numpy array
y_j = y[None,:,:]  # (1, N, 2) numpy array

D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) array of squared distances |x_i-y_j|^2
s_i  = np.argmin(D_ij, axis=1)     # (M,)   array of integer indices
print( s_i[:10] )

###########################################################################
# That's good! Going further, we may speed-up these computations
# using the **CUDA routines** of the PyTorch library:

import torch
use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

x_i = tensor( x[:,None,:] )  # (M, 1, 2) torch tensor
y_j = tensor( y[None,:,:] )  # (1, N, 2) torch tensor

D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) tensor of squared distances |x_i-y_j|^2
s_i  = D_ij.argmin(dim=1)          # (M,)   tensor of integer indices
print( s_i[:10] )

###########################################################################
# But **can we scale to larger point clouds?**
# Unfortunately, tensorized codes will throw an exception
# as soon as the **M-by-N matrix D_ij** stops fitting
# contiguously on the device memory, which typically happens 
# as soon as :math:`\sqrt{MN}` goes past a hardware-dependent threshold 
# in the [5,000; 50,000] range:

M, N = (100000, 200000) if use_cuda else (1000, 2000)
x = np.random.rand(M, 2)
y = np.random.rand(N, 2)

x_i = tensor( x[:,None,:] )  # (M, 1, 2) torch tensor
y_j = tensor( y[None,:,:] )  # (1, N, 2) torch tensor

try:
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) tensor of squared distances |x_i-y_j|^2

except RuntimeError as err: 
    print(err)

###########################################################################
# **That's unfortunate...** And unexpected!
# After all, modern GPUs routinely handle
# `the real-time rendering of scenes with millions of triangles moving around <https://www.youtube.com/watch?v=RaLQuJtQ-gc>`_.
# So how do `SIGGRAPH <https://www.siggraph.org/>`_ programmers achieve such
# a level of performance?
#
# The key to efficient numerical schemes is to remark that
# even though the distance matrix :math:`(D_{i,j})` is not **sparse** in the
# traditional sense, it definitely is **from a computational point of view**.
# As its coefficients are fully described by two lists of points 
# and a **symbolic formula**, sensible implementations will always 
# compute required values on-the-fly...
# Bypassing, **lazily**, the cumbersome pre-computation and storage
# of all pairwise distances :math:`\|x_i-y_j\|^2`.
# 
# 

from pykeops import LazyTensor

x_i = LazyTensor( x[:,None,:] )  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
y_j = LazyTensor( y[None,:,:] )  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y

D_ij = ((x_i - y_j) ** 2).sum(-1)  # **Symbolic** (M, N) matrix of squared distances
print( D_ij )

##########################################################
# With KeOps, implementing **lazy** numerical schemes really
# is **that simple**!
# Our :mod:`LazyTensor <pykeops.common.lazy_tensor.LazyTensor>` variables
# are exactly encoded by a list of raw data arrays plus a custom
# symbolic formula, written with a :doc:`custom mathematical syntax <../../api/math-operations>`
# and modified after each "pythonic" operation such as ``-``, ``**2`` or ``.exp()``.
#
# We may now perform our :func:`.argmin()` reduction using
# an efficient Map-Reduce scheme, implemented 
# as a `highly templated CUDA kernel <https://github.com/getkeops/keops/blob/master/keops/core/GpuConv1D.cu>`_ around
# our custom symbolic formula.
# As evidenced by our :doc:`benchmarks <../../_auto_benchmarks/index>`,
# the KeOps routines have a **linear memory footprint**
# and generally **outperform tensorized GPU implementations by two orders of magnitude**.

s_i  = D_ij.argmin(dim=1).ravel()  # genuine (M,) array of integer indices
print( "s_i is now a {} of shape {}.".format(type(s_i), s_i.shape) )
print( s_i[:10] )

##########################################################
# Going further, you may combine :mod:`LazyTensors <pykeops.common.lazy_tensor.LazyTensor>`
# using a **wide range of mathematical operations**.
# For instance, with data arrays stored directly on the GPU,
# we may compute a Laplacian kernel dot product 
# 
# .. math::
#   a_i ~=~ \sum_{j=1}^N \exp(-\|x_i-y_j\|)\cdot b_j
# 
# in dimension D=10 with:

D = 10
x = torch.randn(M, D).type(tensor)  # M target points in dimension D, stored on the GPU
y = torch.randn(N, D).type(tensor)  # N source points in dimension D, stored on the GPU
b = torch.randn(N, 4).type(tensor)  # N values of the 4D source signal, stored on the GPU

x_i = LazyTensor( x[:,None,:] )  # (M, 1, D) LazyTensor
y_j = LazyTensor( y[None,:,:] )  # (1, N, D) LazyTensor

D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # Symbolic (M, N) matrix of distances
K_ij = (- D_ij).exp()  # Symbolic (M, N) Laplacian (aka. exponential) kernel matrix
a_i = K_ij@b  # The matrix-vector product "@" can be used on "raw" PyTorch tensors!

print("a_i is now a {} of shape {}.".format(type(a_i), a_i.shape) )

#############################################################################
# Automatic differentiation
# -----------------------------------------------
#
# Crucially, :mod:`LazyTensors <pykeops.common.lazy_tensor.LazyTensor>`
# **fully support** the :mod:`torch.autograd` engine:
# you may backprop through a KeOps reduction as easily as through
# a vanilla PyTorch operation.
# For instance, coming back to the kernel dot product above,
# we may compute 



#############################################################################
# Batch processing
# -----------------------------------------------
# Blabla
#


#############################################################################
# Supported formulas
# ------------------------------------
# Blabla
#



