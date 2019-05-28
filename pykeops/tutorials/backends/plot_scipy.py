"""
==========================================
Linking KeOps with scipy.sparse.linalg
==========================================

The `scipy library <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_
provides a simple abstraction for implicit tensors:
the `LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
class,
which represents generic "Matrix-Vector" products
and can be plugged seamlessly in a `large collection <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_ 
of linear algebra routines.

Crucially, KeOps :mod:`LazyTensors <pykeops.LazyTensor>` are now **fully compatible**
with this interface.
As an example, let's see how to combine KeOps with a 
`fast eigenproblem solver <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_ 
to compute **spectral coordinates** on a large 2D point cloud.

.. note::
    Ideally, we'd like to interface KeOps with some
    methods of the `scikit-learn library <https://scikit-learn.org/stable/>`_...
    But this seems out of reach, as most of the sklearn codebase
    relies internally on **explicit numpy arrays**. One day, maybe!

"""

#####################################################################
# Setup 
# -----------------
# Standard imports:
#

import matplotlib.pyplot as plt
import numpy as np

from pykeops import LazyTensor
from pykeops.numpy.utils import IsGpuAvailable
use_cuda = IsGpuAvailable()
dtype = "float32"  # No need for double precision here!


###################################################################
# Create a toy dataset, a spiral in 2D:

N = 10000 if use_cuda else 1000
t = np.linspace(0, 2 * np.pi, N + 1)[:-1]
x = np.stack((.4 + .4 * (t / 7) * np.cos(t), .5 + .3 * np.sin(t)), 1)
x = x + .01 * np.random.randn(*x.shape)
x = x.astype(dtype)

###################################################################
# And display it:
# 
plt.figure(figsize=(8,8))
plt.scatter(x[:,0], x[:,1], s= 5000 / len(x))
plt.axis("equal") ; plt.axis([0,1,0,1])
plt.tight_layout()

#######################################################################
# Spectral coordinates
# -------------------------------
#
# To showcase the potential of the KeOps-SciPy interface,
# we now perform **spectral analysis** on the point cloud **x**.
# As summarized by the `Wikipedia page on spectral clustering <https://en.wikipedia.org/wiki/Spectral_clustering>`_,
# spectral coordinates can be defined as the **eigenvectors** associated
# to the smallest eigenvalues of a `graph Laplacian <https://en.wikipedia.org/wiki/Laplacian_matrix>`_.
#
# When no explicit **adjacency matrix** is available,
# a simple choice is to use a **soft kernel matrix** such as 
# the Gaussian RBF matrix: 
#
# .. math::
#   K_{i,j} ~=~ \exp\big( - \tfrac{1}{2\sigma^2}\|x_i-x_j\|^2 \big),
# 
# which puts
# a smooth link between neighboring points at scale :math:`\sigma`.
#

sigma = .05
x_ = x / sigma
x_i, x_j = LazyTensor( x_[:,None,:] ), LazyTensor( x_[None,:,:] )
K_xx = (- ((x_i - x_j)**2).sum(2) / 2 ).exp()  # Symbolic (N,N) Gaussian kernel matrix

print(K_xx)

########################################################################
# Linear operators
# ~~~~~~~~~~~~~~~~~
#
# As far as **scipy** is concerned, a KeOps :mod:`LazyTensor` such
# as **K_xx** can be directly understood as a 
# `LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_:

from scipy.sparse.linalg import aslinearoperator
K = aslinearoperator( K_xx )

#########################################################
# Just like regular numpy :mod:`arrays` or KeOps :mod:`LazyTensors`,
# :mod:`LinearOperators` fully support the "matrix" product operator ``@``.
# For instance, to compute the mass coefficients
# 
# .. math::
#   D_i ~=~ \sum_{j=1}^N K_{i,j},
#  
# we can simply write:

D = K@np.ones(N, dtype=dtype)  # Sum along the lines of the adjacency matrix


#######################################################################
# Going further, robust and efficient routines such as
# `eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_ 
# can be used to compute the largest (or smallest) eigenvalues of our kernel matrix **K**
# at a reasonable computational cost:
#

from scipy.sparse.linalg import eigsh
eigenvalues, eigenvectors = eigsh( K, k=5 )  # Largest 5 eigenvalues/vectors

print( "Largest eigenvalues:", eigenvalues )
print( "Eigenvectors of shape:", eigenvectors.shape )


############################################
# Graph Laplacian
# ~~~~~~~~~~~~~~~~~~~
# 
# Most importantly, :mod:`LinearOperators` can be composed
# or added with each other.
# To define our implicit **graph Laplacian matrix**:
#
# .. math::
#       L~=~ \text{diag}(D) ~-~ K,
#
# we can simply type:

from scipy.sparse import diags
L = aslinearoperator( diags( D ) ) - K
L.dtype = np.dtype(dtype)  # Scipy Bugfix: by default, "-" removes the dtype information...

##################################################
# Alternatively, we can also use a **symmetric, normalized Laplacian matrix** defined through:
# 
# .. math::
#       L_{\text{norm}}~=~ \text{Id} ~-~ \text{diag}(D^{-1/2}) \,K \,\text{diag}(D^{-1/2}).


from scipy.sparse.linalg.interface import IdentityOperator

D_2 = aslinearoperator( diags( 1 / np.sqrt(D) ) )
L_norm = IdentityOperator( (N,N) ) - D_2@K@D_2
L_norm.dtype = np.dtype(dtype)  # Scipy Bugfix: by default, "-" removes the dtype information...


##################################################
# Then, computing spectral coordinates on **x** is as simple
# as typing:
#

from time import time
start = time()

# Compute the 7 smallest eigenvalues/vectors of our graph Laplacian
eigenvalues, coordinates = eigsh( L , k=7, which="SM" )

print("Smallest eigenvalues of the graph Laplacian, computed in {:.3f}s:".format(time() - start))
print(eigenvalues)

###################################################
# **That's it!**
# As expected, our first eigenvalue is equal to 0,
# up to the convergence of the `Lanczos-like algorithm <https://en.wikipedia.org/wiki/Lanczos_algorithm>`_
# used internally by **eigsh**.
# The spectral coordinates, associated to the **smallest positive eigenvalues**
# of our graph Laplacian, can then be displayed as signals on
# the raw point cloud **x** and be used to perform
# spectral clustering, shape matching or whatever's relevant!

# sphinx_gallery_thumbnail_number = 2
plt.figure(figsize=(12,8))

for i in range(1, 7):
    ax = plt.subplot(2,3,i)
    plt.scatter(x[:,0], x[:,1], c=coordinates[:,i], cmap=plt.cm.Spectral,
                s= 9*500 / len(x))
    ax.set_title( "Eigenvalue {} = {:.2f}".format( i+1, eigenvalues[i] ) )
    plt.axis("equal") ; plt.axis([0,1,0,1])

plt.tight_layout()
plt.show()
