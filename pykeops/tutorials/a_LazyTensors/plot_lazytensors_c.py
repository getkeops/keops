"""
===================================
Advanced usage: symbolic variables
===================================

This tutorial shows how to use the new KeOps helper/container syntax
 
"""

import torch
from pykeops import LazyTensor
use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###########################################################################
# The Vi, Vj, Pm decorators
# -------------------------
# Another way of using the KeOps LazyTensor wrapper is to input 2D tensors, 
# and use the Vi, Vj helpers to specify wether the tensor is to be 
# understood as indexed by i or by j. Here is how it works, if we still
# want to perform the same gaussian convolution. 
#
# We first create the dataset using 2D tensors:

M, N = (100000, 200000) if use_cuda else (1000, 2000)
D = 3

x = torch.randn(M, D).type(tensor)
y = torch.randn(N, D).type(tensor)

############################################################################
# Then we use Vi and Vj to convert to KeOps LazyTensor objects
from pykeops import Vi, Vj, Pm
x_i = Vi(x)  # (M, 1, D) LazyTensor, equivalent to LazyTensor( x[:,None,:] ) 
y_j = Vj(y)  # (1, N, D) LazyTensor, equivalent to LazyTensor( y[None,:,:] ) 

############################################################################
# and perform our operations:
D2xy = ((x_i - y_j) ** 2).sum(-1)
gamma = D2xy.sum(-1).sum_reduction(dim=1)

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
indmin = ((x_i-y_j)**2).sum().argmin(axis=0)

###############################################################################
# Scalar product, absolute value, power operator, and a SoftMax type reduction:
res = (abs(x_i|y_j)**1.5).sumsoftmaxweight(x_i,axis=1)

########################################################################
# The [] operator can be used to do element selection or slicing 
# (Elem or Extract operation in KeOps).
res = (x_i[:2]*y_j[2:]-x_i[2:]*y_j[:2]).sqnorm2().sum(axis=1)

########################################################################
# Kernel inversion : let's do a gaussian kernel inversion. Note that
# we have to use both Vi(x) and Vj(x) here.
# 
e_i = Vi(torch.rand(M,D).type(tensor))
x_j = Vj(x)
D2xx = LazyTensor.sum((x_i-x_j)**2)
sigma = 0.25
Kxx = (-D2xx/sigma**2).exp()
res = LazyTensor.solve(Kxx,e_i,alpha=.1)

#########################################################################
# Use of loops or vector operations for sums of kernels
# -----------------------------------------------------

#############################################################################
# Let us now perform again a kernel convolution, but replacing the gaussian
# kernel by a sum of 4 gaussian kernels with different sigma widths.
# This can be done as follows with a for loop:
sigmas = tensor([0.5, 1.0, 2.0, 4.0])
b_j = Vj(torch.rand(N,D).type(tensor))
Kxy = 0
for sigma in sigmas:
    Kxy += LazyTensor.exp(-D2xy/sigma**2)
gamma = (Kxy*b_j).sum_reduction(axis=1)
 
###############################################################################
# Note again that after the for loop, no actual computation has been performed.
# So we can actually build formulas with much more flexibility than with the 
# use of Genred.
# 
# Ok, this was just to showcase the use of a for loop,
# however in this case there is no need for a for loop, we can do simply:
Kxy = LazyTensor.exp(-D2xy/sigmas**2).sum()
gamma = (Kxy*b_j).sum_reduction(axis=1)

###############################################################################
# This is because all operations are broadcasted, so the / operation above
# works and corresponds to a ./ (scalar-vector element-wise division)




