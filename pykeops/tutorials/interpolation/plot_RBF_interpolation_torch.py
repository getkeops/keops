"""
==================================
Kernel interpolation - PyTorch API
==================================

The :func:`pykeops.torch.KernelSolve` operator allows you to solve optimization
problems of the form

.. math::
    a^{\star}~=~\\arg\min_a \| \,(\\alpha \,\\text{Id}~+~K_{xx})\,a ~-~b\, \|^2_2

where :math:`K_{xx}` is a symmetric, positive definite linear operator
defined through the :ref:`KeOps generic syntax <part.generic_formulas>`
and :math:`\\alpha` is a nonnegative regularisation parameter.
It can thus be used
to solve large-scale `Kriging <https://en.wikipedia.org/wiki/Kriging>`_ 
(aka. `Gaussian process regression <https://scikit-learn.org/stable/modules/gaussian_process.html>`_ )
problems with a **linear memory footprint**.
"""

#############################
# Setup
# ----------------------
#
# Standard imports:

import numpy as np
import torch
import time
from pykeops.torch import Genred
from pykeops.torch import KernelSolve
from matplotlib import pyplot as plt

################################
# Generate some data:

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

N = 10000 if use_cuda else 1000  # Number of samples

# Sampling locations:
x  = torch.rand(N, 1).type(dtype)

# Some random-ish 1D signal:
b  = x + .5 * (6 * x).sin() + .1 * (20 * x).sin() \
       + .05 * torch.randn(N, 1).type(dtype)


#################################
# Kriging in 1D
# ---------------
#
# Specify our **regression model** - a simple **Gaussian** variogram:

formula =  "Exp(- G * SqDist(X,Y) ) * B"  # Gaussian kernel matrix
aliases = ["X = Vx(1)",  # 1st arg: target points, i-variable of size 1
           "Y = Vy(1)",  # 2nd arg: source points, j-variable of size 1
           "B = Vy(1)",  # 3rd arg: source signal, j-variable of size 1
           "G = Pm(1)"]  # 4th arg: scalar parameter, 1/(2*std**2)

###############################################           
# Define an **interpolation problem** by specifying:
# 
# - The kernel computation through **formula**, **aliases** 
#   and the **axis** of reduction.
# - The variable ``B`` with respect to which the computation above is assumed to be **linear**.
# - The ridge regularization parameter **alpha**, which controls the trade-off
#   between a perfect fit (**alpha** = 0) and a 
#   smooth interpolation (**alpha** = :math:`+\infty`).

sigma = .1  # Kernel radius
alpha = 1.  # Ridge regularization

g = torch.Tensor([ .5 / sigma**2 ]).type(dtype)  # RBF bandwidth parameter
Kinv = KernelSolve(formula, aliases, "B", alpha=alpha, axis=1)  # KeOps operator


#################################
# Perform the **Kernel interpolation**:

start = time.time()
a = Kinv(x, x, b, g)
end = time.time()

print('Time to perform an RBF interpolation with {} samples in 1D: {:.5f}s'.format(
       N, end - start) )


#################################
# Display the (fitted) model on the unit interval:
#    

# Extrapolate on a uniform sample:
t = torch.linspace(0,1,1001).type(dtype)[:,None]
K = Genred(formula, aliases, axis=1)
xt = K(t, x, a, g)

# 1D plot:
plt.figure(figsize=(8,6))

plt.scatter(x.cpu()[:, 0], b.cpu()[:, 0], s = 100 / len(x))  # Noisy samples
plt.plot(t.cpu().numpy(), xt.cpu().numpy(), "r")

plt.axis([0,1,0,1]) ; plt.tight_layout() ; plt.show()
 



#################################
# Kriging in 2D
# ---------------
#
# Generate some data:

# Sampling locations:
x  = torch.rand(N, 2).type(dtype)

# Some random-ish 2D signal:
b = ((x - .5) **2).sum(1, keepdim=True)
b[b > .4 **2] = 0
b[b < .3 **2] = 0
b[b >=.3 **2] = 1
b = b + .05 * torch.randn(N, 1).type(dtype)

# Add 10% of outliers:
Nout = N // 10
b[-Nout:] = torch.rand(Nout, 1).type(dtype)

#################################
# Specify our **regression model** - a simple **Exponential** variogram:

formula =  "Exp(- G * Norm2(X-Y) ) * B"  # Laplacian kernel matrix
aliases = ["X = Vx(2)",  # 1st arg: target points, i-variable of size 2
           "Y = Vy(2)",  # 2nd arg: source points, j-variable of size 2
           "B = Vy(1)",  # 3rd arg: source signal, j-variable of size 1
           "G = Pm(1)"]  # 4th arg: scalar parameter, 1/std

###############################################           
# Define an **interpolation problem** by specifying:
# 
# - The kernel computation through **formula**, **aliases** 
#   and the **axis** of reduction.
# - The variable ``B`` with respect to which the computation above is assumed to be **linear**.
# - The ridge regularization parameter **alpha**, which controls the trade-off
#   between a perfect fit (**alpha** = 0) and a 
#   smooth interpolation (**alpha** = :math:`+\infty`).

sigma = .1  # Kernel radius
alpha = 10  # Ridge regularization

g = torch.Tensor([ 1. / sigma ]).type(dtype)  # RBF bandwidth parameter
Kinv = KernelSolve(formula, aliases, "B", alpha=alpha, axis=1)  # KeOps operator


#################################
# Perform the **Kernel interpolation**:

start = time.time()
a = Kinv(x, x, b, g)
end = time.time()

print('Time to perform an RBF interpolation with {} samples in 2D: {:.5f}s'.format(
       N, end - start) )


#################################
# Display the (fitted) model on the unit square:
#    

# Extrapolate on a uniform sample:
X = Y = torch.linspace(0, 1, 101).type(dtype)
X, Y = torch.meshgrid(X, Y)
t = torch.stack( (X.contiguous().view(-1), Y.contiguous().view(-1)), dim=1 )
K = Genred(formula, aliases, axis=1)
xt = K(t, x, a, g).view(101,101)

# 2D plot: noisy samples and interpolation in the background
plt.figure(figsize=(8,8))

plt.scatter(x.cpu()[:, 0], x.cpu()[:, 1], c = b.cpu().view(-1), s = 25000 / len(x), cmap="bwr")
plt.imshow(xt.cpu().numpy()[::-1,:], interpolation="bilinear", extent=[0,1,0,1], cmap="coolwarm")

# sphinx_gallery_thumbnail_number = 2
plt.axis([0,1,0,1]) ; plt.tight_layout() ; plt.show()
 
