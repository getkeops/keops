"""
===================================
Advanced usage: symbolic variables
===================================

This tutorial shows how to use the new KeOps helper/container syntax
 
"""

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
res = keops.solve(Kxx,ei,alpha=.1)

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




