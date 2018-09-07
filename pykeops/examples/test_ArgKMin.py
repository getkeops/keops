"""
Example of KeOps arg-k-min reduction using the generic syntax. 
We define a dataset of N points in R^D, then compute for each
point the indices of its K nearest neighbours (including itself).
This example uses a pure numpy framework (without Pytorch).

"""

#--------------------------------------------------------------#
#                     Standard imports                         #
#--------------------------------------------------------------#
import time
import numpy as np


#--------------------------------------------------------------#
#   Please use the "verbose" compilation mode for debugging    #
#--------------------------------------------------------------#
import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

from pykeops.numpy import Genred

# import pykeops
# pykeops.verbose = False

#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#
N = 5000
D = 2
K = 3

type = 'float32'  # May be 'float32' or 'float64'

x = np.random.randn(N,D).astype(type)

#--------------------------------------------------------------#
#                        Kernel                                #
#--------------------------------------------------------------#
formula = 'SqDist(x,y)'
variables = ['x = Vx('+str(D)+')',  # First arg   : i-variable, of size D
             'y = Vy('+str(D)+')']  # Second arg  : j-variable, of size D

start = time.time()

# The parameter reduction_op='ArgKMin' together with axis=1 means that the reduction operation
# is a sum over the second dimension j. Thence the results will be an i-variable.
my_routine = Genred(formula, variables, reduction_op='ArgKMin', axis=1, cuda_type=type, opt_arg=K)

c = my_routine(x, x, backend="auto").astype(int)

# N.B.: If CUDA is available + backend="auto" (or not specified),
#       KeOps will load the data on the GPU + compute + unload the result back to the CPU,
#       as it is assumed to be more efficient.
#       By specifying backend="CPU", you can make sure that the result is computed
#       using a simple C++ for loop

print("Time to compute the convolution operation: ",round(time.time()-start,5),"s")

print("Output values :")
print(c)
