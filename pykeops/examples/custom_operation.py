# In this demo, we show how to write a (completely) new formula with KeOps.
# Using the low-level "GenericSum/LogSumExp/Max" operators, one can compute
# any formula written as :
#
#    a_i = Reduction( f( p^1,p^2,..., x^1_i,x^2_i,..., y^1_j,y^2_j,...) )
#
# Where:
# - the p^k   's are vector parameters
# - the x^k_i 's are vector variables, indexed by "i"
# - the x^k_i 's are vector variables, indexed by "i"
# - f is an arbitrary function, defined using the "./keops/core" syntax.
# - Reduction is one of :
#   - Sum         (GenericSum)
#   - log-Sum-exp (GenericLogSumExp)
#   - Max         (GenericMax)
#
# Here, given:
# - p,   a vector of size 2
# - x_i, an N-by-D array
# - y_j, an M-by-D array
#
# We will compute (a_i), an N-by-D array given by:
#
#   a_i = sum_{j=1}^M (<x_i,y_j>**2) * ( p[0]*x_i + p[1]*y_j ) 
# 
# N.B.: if you are just interested in writing a new "kernel" formula,
#       you may use the (more convenient) syntax showcased in custom_kernel.py


# Add pykeops to the path
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

# Standard imports
import torch
from torch          import Tensor
from torch.autograd import Variable, grad
from pykeops.torch.generic_sum       import GenericSum
from pykeops.torch.generic_logsumexp import GenericLogSumExp

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor



# Define our dataset
N = 10 ; M = 20 ; D = 3
p = Variable(torch.randn(  2 ),  requires_grad=True).type(dtype)
x = Variable(torch.randn( N,D ), requires_grad=True).type(dtype)
y = Variable(torch.randn( M,D ), requires_grad=True).type(dtype)

# + some random gradient to backprop:
g = Variable(torch.randn( N,D ), requires_grad=True).type(dtype)

# 
aliases = [ "P = Pm(0,2)",                # 1st argument, dim 2, a parameter
            "X = Vx(1," + str(D) + ") ",  # 2nd argument, dim D, indexed by i
            "Y = Vy(2," + str(D) + ") ",  # 3rd argument, dim D, indexed by j
            ]

#           (<x_i,y_j>**2) * (       p[0]*x_i  +       p[1]*y_j )
formula = "Pow( (X,Y) , 2) * ( (Elem(P,0) * X) + (Elem(P,1) * Y) )"

# stands for:       a_i ,    p  ,   x_i ,  y_j  ,
signature   =   [ (D, 0), (2, 2), (D, 0), (D, 1) ]
sum_index   = 0 # the result is indexed by "i"; for "j", use "1"


genconv = GenericSum.apply
backend = "auto"

a  = genconv( backend, aliases, formula, signature, sum_index, p, x, y)

# We can compute gradients wrt all Variables - just like with 
# any other PyTorch operator, really.
# Notice the "create_graph=True", which allows us to compute
# higher order derivatives.
[grad_p, grad_x, grad_y]   = grad( a, [p, x, y], g, create_graph=True)

print(a)