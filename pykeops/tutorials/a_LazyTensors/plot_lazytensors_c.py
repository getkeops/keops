"""
=========================================================
Advanced usage: Vi, Vj, Pm helpers and symbolic variables
=========================================================

This tutorial shows some advanced features of the LazyTensor class.
 
"""

import time

import torch

from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###########################################################################
# The Vi, Vj, Pm decorators
# -------------------------
# This part presents an alternative style for using the KeOps LazyTensor wrapper, that
# some users may find more convenient. The idea is to always input 2D tensors,
# and use the :func:`Vi <pykeops.torch.Vi>`, :func:`Vj <pykeops.torch.Vj>` helpers described below to specify wether the tensor is to be
# understood as indexed by i (i.e. with an equivalent shape of the form (M,1,D))
# or by j (shape of the form (1,N,D)). Note that it is currently not possible to use
# additional batch dimensions with this specific syntax.
#
# Here is how it works, if we
# want to perform a simple gaussian convolution.
#
# We first create the dataset using 2D tensors:

M, N = (100000, 200000) if use_cuda else (1000, 2000)
D = 3

x = torch.randn(M, D).type(tensor)
y = torch.randn(N, D).type(tensor)

############################################################################
# Then we use :func:`Vi <pykeops.torch.Vi>` and :func:`Vj <pykeops.torch.Vj>` to convert to KeOps LazyTensor objects
from pykeops.torch import Vi, Vj

x_i = Vi(x)  # (M, 1, D) LazyTensor, equivalent to LazyTensor( x[:,None,:] )
y_j = Vj(y)  # (1, N, D) LazyTensor, equivalent to LazyTensor( y[None,:,:] )

############################################################################
# and perform our operations:
D2xy = ((x_i - y_j) ** 2).sum()
gamma = D2xy.sum_reduction(dim=1)

#########################################################################
# Note that in the first line we used ``sum`` without any axis or dim parameter.
# This is equivalent to ``sum(-1)`` or ``sum(dim=2)``, because
# the axis parameter is set to ``2`` by default. But speaking about ``dim=2``
# here with the :func:`Vi <pykeops.torch.Vi>`, :func:`Vj <pykeops.torch.Vj>` helpers could be misleading.
# Similarly we used ``sum_reduction`` instead of ``sum`` to make it clear
# that we perform a reduction, but sum and sum_reduction with ``dim=0`` or ``1``
# are equivalent (however ``sum_reduction`` with ``dim=2`` is forbidden)


############################################################################
# We have not spoken about :func:`Pm <pykeops.torch.Pm>` yet. In fact :func:`Pm <pykeops.torch.Pm>` is used to introduce
# scalars or 1D vectors of parameters into formulas, but it is useless
# in such examples because scalars, lists of scalars, 0D or 1D NumPy vectors
# are automatically converted into parameters when combined with
# KeOps formulas. We will have to use :func:`Pm <pykeops.torch.Pm>` in other parts below.


########################################################################
# Other examples
# --------------
# All KeOps operations and reductions
# are available, either via operators or methods. Here are one line
# examples
#
# Getting indices of closest point between x and y:
indmin = ((x_i - y_j) ** 2).sum().argmin(axis=0)

###############################################################################
# Scalar product, absolute value, power operator, and a SoftMax type reduction:
res = (abs(x_i | y_j) ** 1.5).sumsoftmaxweight(x_i, axis=1)

########################################################################
# The ``[]`` operator can be used to do element selection or slicing
# (Elem or Extract operation in KeOps).
res = (x_i[:2] * y_j[2:] - x_i[2:] * y_j[:2]).sqnorm2().sum(axis=1)

########################################################################
# Kernel inversion : let's do a gaussian kernel inversion. Note that
# we have to use both :func:`Vi <pykeops.torch.Vi>` and :func:`Vj <pykeops.torch.Vj>` helpers on the same tensor ``x`` here.
#
e_i = Vi(torch.rand(M, D).type(tensor))
x_j = Vj(x)
D2xx = LazyTensor.sum((x_i - x_j) ** 2)
sigma = 0.25
Kxx = (-D2xx / sigma ** 2).exp()
res = LazyTensor.solve(Kxx, e_i, alpha=0.1)

#########################################################################
# Use of loops or vector operations for sums of kernels
# -----------------------------------------------------

#############################################################################
# Let us now perform again a kernel convolution, but replacing the gaussian
# kernel by a sum of 4 gaussian kernels with different sigma widths.
# This can be done as follows with a for loop:
sigmas = tensor([0.5, 1.0, 2.0, 4.0])
b_j = Vj(torch.rand(N, D).type(tensor))
Kxy = 0
for sigma in sigmas:
    Kxy += LazyTensor.exp(-D2xy / sigma ** 2)
gamma = (Kxy * b_j).sum_reduction(axis=1)

###############################################################################
# Note again that after the for loop, no actual computation has been performed.
# So we can actually build formulas with much more flexibility than with the
# use of Genred.
#
# Ok, this was just to showcase the use of a for loop,
# however in this case there is no need for a for loop, we can do simply:
Kxy = LazyTensor.exp(-D2xy / sigmas ** 2).sum()
gamma = (Kxy * b_j).sum_reduction(axis=1)

###############################################################################
# This is because all operations are broadcasted, so the ``/`` operation above
# works and corresponds to a ``./`` (scalar-vector element-wise division)


###################################################################################
# The "no call" mode
# -----------------------------------------------------
# When using a reduction operation, the user has the choice to actually not perform
# the computation directly and instead output a KeOps object which is
# direclty callable. This can be done using the "call=False" option

M, N = (100000, 200000) if use_cuda else (1000, 2000)
D, Dv = 3, 4

x = torch.randn(M, D).type(tensor)
y = torch.randn(N, D).type(tensor)
b = torch.randn(N, Dv).type(tensor)
sigmas = tensor([0.5, 1.0, 2.0, 4.0])
from pykeops.torch import Vi, Vj, Pm

x_i = Vi(x)  # (M, 1, D) LazyTensor, equivalent to LazyTensor( x[:,None,:] )
y_j = Vj(y)  # (1, N, D) LazyTensor, equivalent to LazyTensor( y[None,:,:] )
b_j = Vj(b)  # (1, N, D) LazyTensor, equivalent to LazyTensor( b[None,:,:] )

D2xy = ((x_i - y_j) ** 2).sum(-1)

Kxy = LazyTensor.exp(-D2xy / sigmas ** 2).sum()

gammafun = (Kxy * b_j).sum_reduction(axis=1, call=False)

###########################################################################
# Here gammafun is a function and can be evaluated later
gamma = gammafun()

###########################################################################
# This is usefull in order to avoid the small overhead
# caused by using the container syntax inside loops if one wants to perform
# a large number of times the same reduction.
# Here is an example where we compare the two approaches on small size data
# to see the effect of the overhead

M, N = 50, 30
x = torch.rand(M, D).type(tensor)
y = torch.rand(N, D).type(tensor)
beta = torch.rand(N, Dv).type(tensor)

xi, yj, bj = Vi(x), Vj(y), Vj(beta)
dxy2 = LazyTensor.sum((xi - yj) ** 2)

Niter = 1000

start = time.time()
for k in range(Niter):
    Kxyb = LazyTensor.exp(-dxy2 / sigmas ** 2).sum() * bj
    gamma = Kxyb.sum_reduction(axis=1)
end = time.time()
print(
    "Timing for {} iterations: {:.5f}s = {} x {:.5f}s".format(
        Niter, end - start, Niter, (end - start) / Niter
    )
)

start = time.time()
Kxyb = LazyTensor.exp(-dxy2 / sigmas ** 2).sum() * bj
gammafun = Kxyb.sum_reduction(axis=1, call=False)
for k in range(Niter):
    gamma = gammafun()
end = time.time()
print(
    "Timing for {} iterations: {:.5f}s = {} x {:.5f}s".format(
        Niter, end - start, Niter, (end - start) / Niter
    )
)

###########################################################################
# Of course this means the user has to perform in-place operations
# over tensors ``x``, ``y,`` ``beta`` inside the loop, otherwise the result of the
# call to ``gammafun`` will always be the same. This is not very convenient,
# so we provide also a "symbolic variables" syntax.

###########################################################################
# Using "symbolic" variables in formulas
# -----------------------------------------------------
#
# Instead of inputing tensors to the :func:`Vi <pykeops.torch.Vi>`, :func:`Vj <pykeops.torch.Vj>`, :func:`Pm <pykeops.torch.Pm>` helpers, one may specify
# the variables as symbolic, providing an index and a dimension:
xi = Vi(0, D)
yj = Vj(1, D)
bj = Vj(2, Dv)
Sigmas = Pm(3, 4)

###########################################################################
# Now we build the formula as before
dxy2 = LazyTensor.sum((xi - yj) ** 2)
Kxyb = LazyTensor.exp(-dxy2 / Sigmas ** 2).sum() * bj
gammafun = Kxyb.sum_reduction(axis=1)

###############################################################################
# Note that we did not have to specify ``call=False`` because since the
# variables are symbolic, no computation can be done of course. So the
# ouput is automatically a function. We can evaluate it by providing the
# arguments in the order specified by the index argument given to :func:`Vi <pykeops.torch.Vi>`, :func:`Vj <pykeops.torch.Vj>`, :func:`Pm <pykeops.torch.Pm>`:
gamma = gammafun(x, y, beta, sigmas)

###########################################################################
# Symbolic and non symbolic variables can be mixed. For example if we want
# to fix ``x``, ``beta`` and ``sigmas`` in the previous example and make the reduction
# a function of ``y`` only we can write:
xi = Vi(x)
yj = Vj(0, D)
bj = Vj(beta)

dxy2 = LazyTensor.sum((xi - yj) ** 2)
Kxyb = LazyTensor.exp(-dxy2 / sigmas ** 2).sum() * bj
gammafun = Kxyb.sum_reduction(axis=1)
print(gammafun)

gamma = gammafun(y)
