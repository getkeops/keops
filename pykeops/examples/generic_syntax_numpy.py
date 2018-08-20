"""
Example of KeOps reduction using the generic syntax. This example
uses a pure numpy framework (without Pytorch).

# this example computes the following tensor operation :
# inputs :
#   x   : a 3000x3 tensor, with entries denoted x_i^u
#   y   : a 5000x3 tensor, with entries denoted y_j^u
#   a   : a 5000x1 tensor, with entries denoted a_j
#   p   : a scalar (entered as a 1x1 tensor)
# output :
#   c   : a 3000x3 tensor, with entries denoted c_i^u, such that
#   c_i^u = sum_j (p-a_j)^2 exp(x_i^u+y_j^u)

"""

#--------------------------------------------------------------#
#                     Standard imports                         #
#--------------------------------------------------------------#
import time
import numpy as np


#--------------------------------------------------------------#
#   Please use the "verbose" compilation mode for debugging    #
#--------------------------------------------------------------#
import os.path, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

from pykeops.numpy import Genred

# import pykeops
# pykeops.verbose = False


#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#
M = 3000
N = 5000

type = 'float32'  # May be 'float32' or 'float64'

x = np.random.randn(M,3).astype(type)
y = np.random.randn(N,3).astype(type)
a = np.random.randn(N,1).astype(type)
p = np.random.randn(1,1).astype(type)


#--------------------------------------------------------------#
#                        Kernel                                #
#--------------------------------------------------------------#
formula = 'Square(p-a)*Exp(x+y)'
variables = ['x = Vx(3)',  # First arg   : i-variable, of size 3
             'y = Vy(3)',  # Second arg  : j-variable, of size 3
             'a = Vy(1)',  # Third arg   : j-variable, of size 1 (scalar)
             'p = Pm(1)']  # Fourth  arg : Parameter,  of size 1 (scalar)


start = time.time()

# The parameter reduction_op='Sum' together with axis=1 means that the reduction operation
# is a sum over the second dimension j. Thence the results will be an i-variable.
my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
c = my_routine(x, y, a, p, backend="auto")

# N.B.: If CUDA is available + backend="auto" (or not specified),
#       KeOps will load the data on the GPU + compute + unload the result back to the CPU,
#       as it is assumed to be more efficient.
#       By specifying backend="CPU", you can make sure that the result is computed
#       using a simple C++ for loop

print("Time to compute the convolution operation: ",round(time.time()-start,5),"s")

#--------------------------------------------------------------#
#                        Gradient                              #
#--------------------------------------------------------------#
# Now, let's compute the gradient of "c" with respect to y. 
# Note that since "c" is not scalar valued, its "gradient" should be understood as 
# the adjoint of the differential operator, i.e. as the linear operator that takes as input 
# a new tensor "e" with same size as "c" and outputs a tensor "g" with same size as "y"
# such that for all variation δy of y :
#    < dc.δy , e >_2  =  < g , δy >_2  =  < δy , dc*.e >_2

# New variable of size Mx3 used as input of the gradient
e = np.random.randn(M, 3).astype(type)

# Thankfully, KeOps provides an autodiff engine for formulas. However, without PyTorch's
# autodiff engine, we need to specify everything by hand: simply add the gradient operator
#  around formula: i.e  Grad( the_formula , variable_to_differentiate, input_of_the_gradient)
formula_grad =  'Grad(' + formula + ', y, e)'

# This new formula has an a new variable (namely the input variable e)
variables_grad = variables + ['e = Vx(3)'] # Fifth  arg : i-variable, of size 3 ... Just like "c" !

my_grad = Genred(formula_grad, variables_grad, reduction_op='Sum', axis=1, cuda_type=type)

start = time.time()
d = my_grad(x, y, a, p, e)
print('Time to compute the gradient of the convolution operation: ', round(time.time()-start,5), 's')
