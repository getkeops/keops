# In this demo, we show how to write a (completely) new formula with KeOps.
#
# Using the low-level "GenericSum/LogSumExp/Max" operators, one can compute
# (with autodiff, without memory overflows) any formula written as :
#
#      a_i = Reduction_j( f( p^1,p^2,..., x^1_i,x^2_i,..., y^1_j,y^2_j,...) )
# or   b_j = Reduction_i( f( p^1,p^2,..., x^1_i,x^2_i,..., y^1_j,y^2_j,...) )
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
#
# In this demo file, given:
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


def my_formula(p, x, y, backend = "auto") :
    """
    Applies a custom formula on the torch variables P, X and Y.
    Two backends are provided, so that we can check the correctness
    of both implementations.
    """
    if backend == "pytorch" : # Vanilla PyTorch implementation ===================
        scals = (x @ y.t())**2 # Memory-intensive computation!
        a = p[0] * scals.sum(1).view(-1,1) * x \
          + p[1] * (scals @ y)
        return a

    else : # KeOps implementation ================================================
        # We now expose the low-level syntax of KeOps.
        # The library relies on vector "Variables" which can be either:
        # - indexed by "i" ("x" variables, category 0)
        # - indexed by "j" ("y" variables, category 1)
        # - constant across the reduction ("parameters", category 2)
        #
        # First of all, we must define a "who's who" list of the variables used,
        # by specifying their categories, index in the arguments' list, and dimensions:
        aliases = [ "P = Pm(0,2)",                         #  a parameter, 1st argument, dim 2. 
                    "X = Vx(1," + str(x.shape[1]) + ") ",  # indexed by i, 2nd argument, dim D.
                    "Y = Vy(2," + str(y.shape[1]) + ") ",  # indexed by j, 3rd argument, dim D.
                  ]

        # The actual formula:
        # a_i   =   (<x_i,y_j>**2) * (       p[0]*x_i  +       p[1]*y_j )
        formula = "Pow( (X,Y) , 2) * ( (Elem(P,0) * X) + (Elem(P,1) * Y) )"

        # KeOps expects from the low-level hacker an explicit "signature"
        # that specifies the dimensions and categories of the variables
        # used by the custom formula.
        # (strong typing should help debugging)
        #
        # In practice, we provide a list of (dimension,category) integer pairs,
        # sorted in the [ output,  arg_0,  arg_1,   ...  ] order.
        # Here, we use:     a_i ,    p  ,   x_i ,   y_j  .
        signature   =   [ (D, 0), (2, 2), (D, 0), (D, 1) ]

        # Finally, we specify if the reduction should be done wrt. "i" or "j".
        sum_index   = 0 # the result is indexed by "i"; for "j", use "1"

        genconv = GenericSum.apply
        a  = genconv( backend, aliases, formula, signature, sum_index, p, x, y)
        return a


# Test ========================================================================
# Define our dataset
N = 1000 ; M = 2000 ; D = 3
# PyTorch tip: do not "require_grad" of "x" if you do not intend to
#              actually compute a gradient wrt. said variable "x".
#              Given this info, PyTorch (+ KeOps) is smart enough to
#              skip the computation of unneeded gradients.
p = Variable(torch.randn(  2  ), requires_grad=True ).type(dtype)
x = Variable(torch.randn( N,D ), requires_grad=False).type(dtype)
y = Variable(torch.randn( M,D ), requires_grad=True ).type(dtype)

# + some random gradient to backprop:
g = Variable(torch.randn( N,D ), requires_grad=True ).type(dtype)

for backend in ["pytorch", "auto"] :
    print("Backend :", backend, "============================" )
    a = my_formula(p, x, y, backend=backend)

    # We can compute gradients wrt all Variables - just like with 
    # any other PyTorch operator, really.
    # Notice the "create_graph=True", which allows us to compute
    # higher order derivatives if needed.
    [grad_p, grad_y]   = grad( a, [p, y], g, create_graph=True)

    print("(a_i) :", a[:3,:])
    print("(∂_p a).g :", grad_p )
    print("(∂_y a).g :", grad_y[:3,:])