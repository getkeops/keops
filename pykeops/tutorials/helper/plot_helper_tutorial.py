"""
===============================
Helper/Container syntax
===============================

This tutorial shows how to use the new KeOps helper/container syntax
 
"""

#########################################################################
# Setup 
# -----------------
# Standard imports:

import numpy as np
import torch
from pykeops import LazyTensor as keops
from pykeops import Vi, Vj, Pm
import time

#########################################################################
# The 3D tensor syntax, aka NumPy or PyTorch style
# ------------------------------------------------

#############################################################################
# Let us first compute a simple gaussian convolution: let
#
#  - :math:`x_i` be arbistrary 2D points, :math:`1\leq i\leq M`
#  - :math:`y_j` be arbistrary 2D points, :math:`1\leq j\leq N`
#  - :math:`\beta_j` be arbistrary 4D vectors, :math:`1\leq j\leq N`
#  - :math:`\sigma` as scale parameter
#
# We would like to compute, for :math:`1\leq i\leq M`,
#
# .. math::
#   \gamma_i ~=~ \sum_{j=1}^N\exp\left(-\frac{\|x_i-y_j\|^2}{\sigma^2}\right)\beta_j



################################################################
# Here is the corresponding computation using KeOps:
#
# We first define the dataset. In this first part we will use
# a "3D" tensor convention : the i indices correspond to the
# first axis, the j indices to the 2nd axis, and the 3rd axis
# is used for the dimension of points or vectors
M, N = 50, 30
D, Dv = 4, 2
x = np.random.rand(M,1,D)
y = np.random.rand(1,N,D)
beta = np.random.rand(1,N,Dv)
sigma = 0.25

################################################################
# We first convert tensors x,y,b into KeOps objects:
X, Y, Beta = keops(x), keops(y), keops(beta)

########################################################################
# Then we perform operations as if we were using usual 3D tensors, 
# using usual broadcasting conventions as in NumPy.
dxy2 = keops.sum((X-Y)**2,axis=2)
Kxyb = keops.exp(-dxy2/sigma**2) * Beta

########################################################################
# At this point we have to think about Kxyb as having shape (M,N,Dv)
# although in fact it is just a KeOps object and no computation has been
# performed yet. We can check its formula and shape using print:
print(Kxyb)

########################################################################
# Finally we compute the final sum over the j
gamma = Kxyb.sum(axis=1)

########################################################################
# Now since we have called a reduction operation over the second axis
# (i.e. over the j indices), the KeOps engine has been called and the
# output gamma is not any more a KeOps object but a NumPy tensor of 
# shape (M,Dv).

#######################################################################
# Let us do it again, starting from PyTorch tensors, and computing
# a gradient to see how the autodiff goes through.
# Note that we use "dim" instead of "axis" keywords
# to mimic PyTorch conventions
# but the two keys are equivalent in KeOps.
x = torch.rand(M,1,D,requires_grad=True)
y = torch.rand(1,N,D)
beta = torch.rand(1,N,Dv)

X, Y, Beta = keops(x), keops(y), keops(beta)
dxy2 = keops.sum((X-Y)**2,dim=2)
Kxyb = keops.exp(-dxy2/sigma**2) * Beta
gamma = Kxyb.sum(dim=1)

e = torch.rand(M,Dv)
grad_gamma = torch.autograd.grad(gamma,x,e)

#######################################################################
# Note that we can use also a "grad" method, which takes the
# symbolic gradient of the formula before doing the reduction.
# This gives the same result
E = keops(e,axis=0)
grad_gamma_ = Kxyb.grad(X,E).sum(dim=1)

#########################################################################
# The Vi, Vj, Pm style
# ---------------------
# Another way of using the KeOps container syntax is to use 2D tensors, 
# and use the Vi, Vj helpers to specify wether the tensor is to be 
# understood as indexed by i or by j. Here is how it works, if we still
# want to perform the same gaussian convolution. 
#
# We first create the dataset using 2D tensors:
x = np.random.rand(M,D)
y = np.random.rand(N,D)
beta = np.random.rand(N,Dv)
sigma = 0.25

#########################################################################
# Then we use Vi and Vj to convert to KeOps objects
xi, yj, bj = Vi(x), Vj(y), Vj(beta)

#########################################################################
# and perform the operations:
dxy2 = keops.sum((xi-yj)**2)
Kxyb = keops.exp(-dxy2/sigma**2) * bj
gamma = Kxyb.sum_reduction(axis=1)

#########################################################################
# Note that in the first line we used "sum" without any axis parameter.
# This is just completely equivalent with the initial example, because
# the axis parameter is set to 2 by default. But speaking about axis=2
# here with the Vi, Vj helpers could be misleading for the user.
# Similarly we used "sum_reduction" instead of "sum" to make it clear
# that we perform a reduction, but sum and sum_reduction with axis=0 or 1
# are equivalent (but sum_reduction with axis=2 is forbidden)


############################################################################
# We have not spoken about Pm yet. In fact Pm is used to introduce 
# scalars or 1D vectors of parameters into formulas, but it is useless
# in such examples because scalars, lists of scalars, 0D or 1D NumPy vectors
# are automatically converted into parameters when combined with 
# KeOps formulas. We will have to use Pm in other parts below.


########################################################################
# Other examples
# --------------
# All KeOps operations and reductions
# are available, either via operators or methods. Here are one line
# examples 
# 
# Getting indices of closest point between x and y:
indmin = ((xi-yj)**2).sum().argmin(axis=0)

###############################################################################
# Scalar product, absolute value, power operator, and a SoftMax type reduction:
res = (abs(xi|yj)**1.5).sumsoftmaxweight(xi,axis=1)

########################################################################
# The [] operator can be used to do element selection or slicing 
# (Elem or Extract operation in KeOps).
res = (xi[:2]*yj[2:]-xi[2:]*yj[:2]).sqnorm2().sum(axis=1)

########################################################################
# Kernel inversion : let's do a gaussian kernel inversion. Note that
# we have to use both Vi(x) and Vj(x) here.
# 
ei = Vi(np.random.rand(M,Dv))
xj = Vj(x)
dx2 = keops.sum((xi-xj)**2)
Kxx = (-dx2/sigma**2).exp()
res = keops.kernelsolve(Kxx,ei,alpha=.1)

#########################################################################
# Use of loops or vector operations for sums of kernels
# -----------------------------------------------------

#############################################################################
# Let us now perform again a kernel convolution, but replacing the gaussian
# kernel by a sum of 4 gaussian kernels with different sigma widths.
# This can be done as follows with a for loop:
sigmas = np.array([0.5, 1.0, 2.0, 4.0])
Kxy = 0
for sigma in sigmas:
    Kxy += keops.exp(-dxy2/sigma**2)
gamma = (Kxy*bj).sum_reduction(axis=1)
 
###############################################################################
# Note again that after the for loop, no actual computation has been performed.
# So we can actually build formulas with much more flexibility than with the 
# use of Genred.
# 
# Ok, this was just to showcase the use of a for loop,
# however in this case there is no need for a for loop, we can do simply:
Kxy = keops.exp(-dxy2/sigmas**2).sum()
gamma = (Kxy*bj).sum_reduction(axis=1)

###############################################################################
# This is because all operations are broadcasted, so the / operation above
# works and corresponds to a ./ (scalar-vector element-wise division)

###################################################################################
# The "no call" mode
# -----------------------------------------------------
# When using a reduction operation, the user has the choice to actually not perform
# the computation directly and instead output a KeOps object which is
# direclty callable. This can be done using the "call=False" option
gammafun = (Kxy*bj).sum_reduction(axis=1,call=False)

###########################################################################
# Here gammafun is a function and can be evaluated later
gamma = gammafun()

###########################################################################
# This is usefull in order to avoid the small overhead
# caused by using the container syntax inside loops if one wants to perform
# a large number of times the same reduction.
# Here is an example where we compare the two approaches:

Niter = 1000

start = time.time()
for k in range(Niter):
    Kxyb = keops.exp(-dxy2/sigmas**2).sum() * bj
    gamma = Kxyb.sum_reduction(axis=1)
end = time.time()
print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s'.format(
    Niter, end - start, Niter, (end - start) / Niter))

start = time.time()
Kxyb = keops.exp(-dxy2/sigmas**2).sum() * bj
gammafun = Kxyb.sum_reduction(axis=1,call=False)
for k in range(Niter):
    gamma = gammafun()
end = time.time()
print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s'.format(
    Niter, end - start, Niter, (end - start) / Niter))

###########################################################################
# Of course this means the user has to perform in-place operations
# over tensors x, y, beta inside the loop, otherwise the result of the
# call to gammafun will always be the same. This is not very convenient,
# so we provide also a "symbolic variables" syntax (see next section)

###########################################################################
# Using "symbolic" variables in formulas
# -----------------------------------------------------
#
# Instead of inputing tensors to the Vi, Vj, Pm helpers, one may specify
# the variables as symbolic, providing an index and a dimension:
xi = Vi(0,D)
yj = Vj(1,D)
bj = Vj(2,Dv)
Sigmas = Pm(3,4)

###########################################################################
# Now we build the formula as before
dxy2 = keops.sum((xi-yj)**2)
Kxyb = keops.exp(-dxy2/Sigmas**2).sum() * bj
gammafun = Kxyb.sum_reduction(axis=1)

###############################################################################
# Note that we did not have to specify "call=False" because since the
# variables are symbolic, no computation can be done of course. So the
# ouput is automatically a function. We can evaluate it by providing the
# arguments in the order specified by the index argument given to Vi, Vj, Pm:
gamma = gammafun(x,y,beta,sigmas)

###########################################################################
# Symbolic and non symbolic variables can be mixed. For example if we want
# to fix x, beta and sigmas in the previous example and make the reduction
# a function of y only we can write:
xi = Vi(x)
yj = Vj(0,D)
bj = Vj(beta)

dxy2 = keops.sum((xi-yj)**2)
Kxyb = keops.exp(-dxy2/sigmas**2).sum() * bj
gammafun = Kxyb.sum_reduction(axis=1)
print(gammafun)

gamma = gammafun(y)




