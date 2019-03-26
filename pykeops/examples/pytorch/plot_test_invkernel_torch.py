"""
KernelSolve reduction
===========================

Let's see how to solve discrete deconvolution problems
using the **conjugate gradient solver** provided by
:func:`pykeops.torch.KernelSolve`.
"""

###############################################################################
# Setup
# ----------------
#
# Standard imports:
#

import torch
import time 

from pykeops.torch import KernelSolve


###############################################################################
# Define our dataset:
#

N  = 5000   # Number of points
D  = 2      # Dimension of the ambient space
Dv = 2      # Dimension of the vectors (= number of linear problems to solve)
sigma = .1  # Radius of our RBF kernel    

x = torch.rand(N, D, requires_grad=True)
b = torch.rand(N, Dv)
g = torch.Tensor([ .5 / sigma**2])  # Parameter of the Gaussian RBF kernel

###############################################################################
# KeOps kernel
# ---------------
#
# Define a Gaussian RBF kernel:
#

formula = 'Exp(- g * SqDist(x,y)) * b'
aliases = ['x = Vi(' + str(D) + ')',   # First arg:  i-variable of size D
           'y = Vj(' + str(D) + ')',   # Second arg: j-variable of size D
           'b = Vj(' + str(Dv) + ')',  # Third arg:  j-variable of size Dv
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
# Compare with a straightforward PyTorch implementation:
#

start = time.time()
K_xx = alpha * torch.eye(N) + torch.exp( -torch.sum( (x[:,None,:] - x[None,:,:])**2,dim=2) / (2*sigma**2) )
c_py = torch.gesv(b, K_xx)[0]
end = time.time()
print('Timing (PyTorch implementation):', round(end - start, 5), 's')
print("Relative error = ",(torch.norm(c - c_py) / torch.norm(c_py)).item())


###############################################################################
# Compare the derivatives:
#


print("1st order derivative")
e = torch.randn(N,D)
start = time.time()
u, = torch.autograd.grad(c, x, e)
end = time.time()
print('Timing (KeOps derivative):', round(end - start, 5), 's')
start = time.time()
u_py, = torch.autograd.grad(c_py, x, e)
end = time.time()
print('Timing (PyTorch derivative):', round(end - start, 5), 's')
print("Relarive error = ",(torch.norm(u - u_py) / torch.norm(u_py)).item())



