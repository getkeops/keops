"""
==========================================
Linking KeOps with scipy.sparse.linalg
==========================================

The `scipy library <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_
provides a nice abstraction for implicit tensors:
the `LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
class,
which represents abstract "Matrix-Vector" products
and can be plugged seamlessly in a `large collection <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_ 
of linear algebra routines.

.. note::
    Ideally, we'd like to interface KeOps with some
    methods of the `scikit-learn library <https://scikit-learn.org/stable/>`_...
    But this seems out of reach. One day, maybe!

"""

#####################################################################
# Setup 
# -----------------
# Standard imports:
#

import matplotlib.pyplot as plt
import numpy as np
from pykeops import LazyTensor

N, sigma = 100, .05

t = np.linspace(0, 2 * np.pi, N + 1)[:-1]
x = np.stack((.4 + .4 * (t / 7) * np.cos(t), .5 + .3 * np.sin(t)), 1)
x = x + .01 * np.random.randn(*x.shape)
x = x.astype(dtype)

x_ = x / sigma
x_i, x_j = LazyTensor( x_[:,None,:] ), LazyTensor( x_[None,:,:] )
K_xx = (- ((x_i - x_j)**2).sum(2) / 2 ).exp()  # Symbolic (N,N) Gaussian kernel matrix

from scipy.sparse.linalg import eigs, eigsh, aslinearoperator

K = aslinearoperator( K_xx )

eigenvalues, eigenvectors = eigsh( K, k=5 )
print(eigenvalues)
print( eigenvectors.shape)


#######################################################################
# Spectral coordinates
# -------------------------------
#
# To illustrate !!!,
# `spectral clustering <https://en.wikipedia.org/wiki/Spectral_clustering>`_

from scipy.sparse import identity, diags
from scipy.sparse.linalg.interface import IdentityOperator

D = K @ np.ones(N)  # Sum along the lines of the adjacency matrix

# Graph Laplacian matrix:
L = aslinearoperator( diags( D ) ) - K

# Symmetric normalized Laplacian matrix:
D_2 = aslinearoperator( diags( 1 / np.sqrt(D) ) )
L_norm = IdentityOperator((N, N)) - D_2 @ K @ D_2

eigenvalues, coordinates = eigsh( aslinearoperator( L ), k=6, which="SM" )
print(eigenvalues)
print( coordinates.shape)

# eigenvalues, coordinates = eigenvalues[::-1], coordinates[:,::-1]

plt.figure()
plt.plot(coordinates[:,0])

plt.figure(figsize=(12,8))

for i in range(6):
    ax = plt.subplot(2,3,i+1)
    plt.scatter(x[:,0], x[:,1], c=coordinates[:,i], cmap=plt.cm.Spectral,
                s= 500 / len(x))
    ax.set_title( "Eigenvalue {} = {:.2e}".format( i+1, eigenvalues[i] ) )

plt.show()
