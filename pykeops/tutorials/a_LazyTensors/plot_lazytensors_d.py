
import time
import numpy as np
import torch
from pykeops import LazyTensor
use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

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
from pykeops import Vi, Vj, Pm
x_i = Vi(x)  # (M, 1, D) LazyTensor, equivalent to LazyTensor( x[:,None,:] ) 
y_j = Vj(y)  # (1, N, D) LazyTensor, equivalent to LazyTensor( y[None,:,:] ) 
b_j = Vj(b)  # (1, N, D) LazyTensor, equivalent to LazyTensor( b[None,:,:] ) 

D2xy = ((x_i - y_j) ** 2).sum(-1)

Kxy = LazyTensor.exp(-D2xy/sigmas**2).sum()

gammafun = (Kxy*b_j).sum_reduction(axis=1,call=False)

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
x = torch.rand(M,D).type(tensor)
y = torch.rand(N,D).type(tensor)
beta = torch.rand(N,Dv).type(tensor)

xi, yj, bj = Vi(x), Vj(y), Vj(beta)
dxy2 = LazyTensor.sum((xi-yj)**2)

Niter = 1000

start = time.time()
for k in range(Niter):
    Kxyb = LazyTensor.exp(-dxy2/sigmas**2).sum() * bj
    gamma = Kxyb.sum_reduction(axis=1)
end = time.time()
print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s'.format(
    Niter, end - start, Niter, (end - start) / Niter))

start = time.time()
Kxyb = LazyTensor.exp(-dxy2/sigmas**2).sum() * bj
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
dxy2 = LazyTensor.sum((xi-yj)**2)
Kxyb = LazyTensor.exp(-dxy2/Sigmas**2).sum() * bj
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

dxy2 = LazyTensor.sum((xi-yj)**2)
Kxyb = LazyTensor.exp(-dxy2/sigmas**2).sum() * bj
gammafun = Kxyb.sum_reduction(axis=1)
print(gammafun)

gamma = gammafun(y)
