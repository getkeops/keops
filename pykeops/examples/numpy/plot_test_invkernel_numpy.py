"""
KernelSolve reduction
===========================

Let's see how to solve discrete deconvolution problems
using the **conjugate gradient solver** provided by
:func:`pykeops.numpy.KernelSolve`.
"""

###############################################################################
# Setup
# ----------------
#
# Standard imports:
#

import numpy as np
import time 

from pykeops.numpy import KernelSolve

###############################################################################
# Define our dataset:
#

N  = 5000   # Number of points
D  = 2      # Dimension of the ambient space
Dv = 2      # Dimension of the vectors (= number of linear problems to solve)
sigma = .1  # Radius of our RBF kernel    

x = np.random.rand(N, D)
b = np.random.rand(N, D)
g = np.array([ .5 / sigma**2])  # Parameter of the Gaussian RBF kernel

###############################################################################
# KeOps kernel
# ---------------
#
# Define a Gaussian RBF kernel:
#

formula = 'Exp(- g * SqDist(x,y)) * b'
aliases = ['x = Vx(' + str(D) + ')',   # First arg:  i-variable of size D
           'y = Vy(' + str(D) + ')',   # Second arg: j-variable of size D
           'b = Vy(' + str(Dv) + ')',  # Third arg:  j-variable of size Dv
           'g = Pm(1)']                # Fourth arg: scalar parameter
             

###############################################################################
# Define the inverse kernel operation, with a ridge regularization **alpha**:
# 

alpha = 0.01
Kinv = KernelSolve(formula, aliases, "b", alpha=alpha, axis=1)

###############################################################################
# .. note::
#   This operator uses a conjugate gradient solver and assumes
#   that **formula** defines a **symmetric**, positive and definite
#   **linear** reduction with respect to the alias ``"b"``
#   specified trough the third argument.
#
# Apply our solver on arbitrary point clouds:
#

print("Solving a Gaussian linear system, with {} points in dimension {}.".format(N,D))
start = time.time()
c = Kinv(x, x, b, g)
end = time.time()
print('Timing (KeOps implementation):', round(end - start, 5), 's')

###############################################################################
# Compare with a straightforward Numpy implementation:
#

start = time.time()
K_xx = alpha * np.eye(N) + np.exp( - g * np.sum( (x[:,None,:] - x[None,:,:]) **2, axis=2) )
c_np = np.linalg.solve( K_xx, b)
end = time.time()
print('Timing (Numpy implementation):', round(end - start, 5), 's')
print("Relative error = ", np.linalg.norm(c - c_np) / np.linalg.norm(c_np))

