"""
==========================================
Fancy reductions, solving linear systems
==========================================


"""

#############################################################
# As discussed in the previous notebook, 
# KeOps :mod:`LazyTensors <pykeops.common.lazy_tensor.LazyTensor>`
# support a wide range of **mathematical formulas**.
# Let us now discuss the different operators that may be used
# to **reduce** our large M-by-N symbolic tensors into
# vanilla NumPy arrays or PyTorch tensors.
#
# .. note::
#   In this tutorial, we stick to the **PyTorch** interface
#   but note that apart from a few lines on backpropagation,
#   everything here can be seamlessly translated to vanilla NumPy+KeOps code.
#
# LogSumExp, KMin and advanced reductions
# ---------------------------------------------------
#
# First, let's build some large :mod:`LazyTensors <pykeops.common.lazy_tensor.LazyTensor>`
# ``S_ij`` and ``V_ij`` which respectively handle **scalar** and **vector**
# formulas:

import torch
from pykeops import LazyTensor
use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
M, N = (100000, 200000) if use_cuda else (1000, 2000)
D = 3

x = torch.randn(M, D).type(tensor)
y = torch.randn(N, D).type(tensor)

x_i = LazyTensor( x[:,None,:] )  # (M, 1, D) LazyTensor
y_j = LazyTensor( y[None,:,:] )  # (1, N, D) LazyTensor

V_ij = (x_i - y_j)          # (M, N, D) symbolic tensor of differences
S_ij = (V_ij ** 2).sum(-1)  # (M, N, 1) = (M, N) symbolic matrix of squared distances

print(S_ij)
print(V_ij)

############################################################################
# As we've seen earlier, the :meth:`.sum()` reduction can be used
# on both ``S_ij`` and ``V_ij`` to produce genuine PyTorch 2D tensors:

print( "Sum reduction of S_ij wrt. the 'N' dimension:", S_ij.sum(dim=1).shape )

###############################################
# Note that :mod:`LazyTensors <pykeops.common.lazy_tensor.LazyTensor>`
# support reductions over both indexing "M" and "N" dimensions,
# which may be specified using the PyTorch-friendly **dim**
# or the standard NumPy **axis** optional arguments:

print( "Sum reduction of V_ij wrt. the 'M' dimension:", V_ij.sum(axis=0).shape )

##################################################################
# Just like PyTorch tensors, 
# :mod:`LazyTensors <pykeops.common.lazy_tensor.LazyTensor>`
# also support a **stabilized** `log-sum-exp reduction <https://en.wikipedia.org/wiki/LogSumExp>`_,
# computed efficiently with a **running maximum** in the CUDA loop: 

print( "LogSumExp reduction of S_ij wrt. the 'M' dimension:", S_ij.logsumexp(dim=0).shape )

# N.B.: logsumexp reductions are not yet available for vector formulas...
#       but this will be done soon!
# print( "LogSumExp reduction of V_ij wrt. the 'N' dimension:", V_ij.logsumexp(dim=1).shape )


###############################################################
# Going further, the :meth:`.min()`, :meth:`.max()`, :meth:`.argmin()` or :meth:`.argmax()`
# reductions work as expected, following the (sensible) NumPy convention:

print( "Min    reduction of S_ij wrt. the 'M' dimension:", S_ij.min(dim=0).shape )
print( "ArgMin reduction of S_ij wrt. the 'N' dimension:", S_ij.argmin(dim=1).shape )

print( "Max    reduction of V_ij wrt. the 'M' dimension:", V_ij.max(dim=0).shape )
print( "ArgMax reduction of V_ij wrt. the 'N' dimension:", V_ij.argmax(dim=1).shape )

##################################################################
# To compute both quantities in a single pass, feel free to use 
# the :meth:`.min_argmin()` and :meth:`.max_argmax()` reductions:

m_i, s_i = S_ij.min_argmin(dim=0)
print( "Min-ArgMin reduction on S_ij wrt. the 'M' dimension:", m_i.shape, s_i.shape )

m_i, s_i = V_ij.max_argmax(dim=1)
print( "Max-ArgMax reduction on V_ij wrt. the 'N' dimension:", m_i.shape, s_i.shape )

##################################################################
# More interestingly, KeOps also provides support for
# the :meth:`.Kmin()`, :meth:`.argKmin()` and :meth:`.Kmin_argKmin()`
# reductions that may be used to implement an efficient
# :doc:`K-nearest neighbor algorithm <../knn/plot_knn_torch>` :
#

K = 5
print( "KMin    reduction of S_ij wrt. the 'M' dimension:", S_ij.Kmin(K=K, dim=0).shape )
print( "ArgKMin reduction of S_ij wrt. the 'N' dimension:", S_ij.argKmin(K=K, dim=1).shape )

##################################################################
# It even works on vector formulas!

K = 7
print( "KMin    reduction of V_ij wrt. the 'M' dimension:", V_ij.Kmin(K=K, dim=0).shape )
print( "ArgKMin reduction of V_ij wrt. the 'N' dimension:", V_ij.argKmin(K=K, dim=1).shape )

#################################################################
# Finally, the :mod:`.sumsoftmaxweight(weights)` reduction
# may be used  !!!


#################################################################
# Solving linear systems
# -------------------------------------
#
# When working with
#

#################################################################
# .. warning::
#   Blabla kernel