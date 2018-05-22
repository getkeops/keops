"""
Example of KeOps reduction using the generic syntax. This example corresponds
to the one described in the documentation file generic_syntax.md, using
pure numpy framework (without Pytorch).

# this example computes the following tensor operation :
# inputs :
#   p   : a scalar (antered as a 1x1 tensor)
#   a   : a 5000x1 tensor, with entries denoted a_j 
#   x   : a 3000x3 tensor, with entries denoted x_i^u
#   y   : a 5000x3 tensor, with entries denoted y_j^u
# output :
#   c   : a 3000x3 tensor, with entries denoted c_i^u, such that
#   c_i^u = sum_j (p-y_j)^2 exp(a_i^u+b_j^u)

"""

#--------------------------------------------------------------#
#                     Standard imports                         #
#--------------------------------------------------------------#
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import time
import numpy as np

from pykeops.numpy import generic_sum_np


#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#

N = 3000
M = 5000

p = np.random.randn(1,1).astype('float32')
a = np.random.randn(M,1).astype('float32')
x = np.random.randn(N,3).astype('float32')
y = np.random.randn(M,3).astype('float32')


#--------------------------------------------------------------#
#                        Kernel                                #
#--------------------------------------------------------------#

formula =  "Square(p-a)*Exp(x+y)"
types   = ["output = Vx(3)",  # The result is indexed by "i", of size 3.
                "p = Pm(1)",  # First  arg : Parameter,  of size 1 (scalar)
				"a = Vy(1)",  # Second arg : j-variable, of size 1 (scalar)
				"x = Vx(3)",  # Third  arg : i-variable, of size 3
				"y = Vy(3)" ] # Fourth arg : j-variable, of size 3


start = time.time()

my_routine = generic_sum_np(formula, *types)
c = my_routine(p,a,x,y, backend="auto")

# N.B.: If CUDA is available + backend="auto" (or not specified) + the arrays are large enough,
#       KeOps will load the data on the GPU + compute + unload the result back to the CPU,
#       as it is assumed to be more efficient.
#       By specifying backend="CPU", you can make sure that the result is computed
#       using a simple C++ for loop

print("Time to compute the convolution operation: ",round(time.time()-start,2),"s")

#--------------------------------------------------------------#
#                        Gradient                              #
#--------------------------------------------------------------#

# Now, let's compute the gradient of "c" with respect to y. 
# Note that since "c" is not scalar valued, its "gradient" should be understood as 
# the adjoint of the differential operator, i.e. as the linear operator that takes as input 
# a new tensor "e" with same size as "c" and outputs a tensor "g" with same size as "y"
# such that for all variation δy of y :
#    < dc.δy , e >_2  =  < g , δy >_2  =  < δy , dc*.e >_2

# New variable of size Nx3 used as input of the gradient
e = np.random.randn(N,3).astype('float32')

# Thankfully, KeOps provides an autodiff engine for formulas !
formula =  "Grad( Square(p-a)*Exp(x+y), y, e)"

# However, without PyTorch's autodiff engine, we need to specify the Grad's signature by hand:
types   = ["output = Vy(3)",  # The result is indexed by "j", of size 3... Just like "y" !
                "p = Pm(1)",  # First  arg : Parameter,  of size 1 (scalar)
				"a = Vy(1)",  # Second arg : j-variable, of size 1 (scalar)
				"x = Vx(3)",  # Third  arg : i-variable, of size 3
				"y = Vy(3)",  # Fourth arg : j-variable, of size 3
                "e = Vx(3)" ] # Fifth  arg : i-variable, of size 3 ... Just like "c" !

my_grad = generic_sum_np(formula, *types)

start = time.time()
d = my_grad( p,a,x,y,e )
print("Time to compute the gradient of the convolution operation : ",round(time.time()-start,2),"s")
