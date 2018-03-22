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


def my_formula(p, g, x, y, b) :
    aliases = [ #"P = Pm(4," + str(p.shape[0]) + ") ",   
                "G = Vy(0," + str(g.shape[1]) + ") ",                         
                "X = Vx(1," + str(x.shape[1]) + ") ",
                "Y = Vy(2," + str(y.shape[1]) + ") ",
                "B = Vy(3," + str(b.shape[1]) + ") ",
                ]

    formula = "(Exp(-(WeightedSqDist(G,X,Y)))*B)"
    C,D,E = g.shape[1], x.shape[1], b.shape[1]

    signature   =   [ (E, 0), (C, 1), (D, 0), (D, 1), (E, 1)]#, (p.shape[0],2) ]

    # Finally, we specify if the reduction should be done wrt. "i" or "j".
    sum_index   = 0 # the result is indexed by "i"; for "j", use "1"

    genconv = GenericSum.apply
    a  = genconv( "auto", aliases, formula, signature, sum_index, g, x, y, b)#, p)
    return a


# Test ========================================================================
# Define our dataset
N = 1000 ; M = 2000 ; D = 3
# PyTorch tip: do not "require_grad" of "x" if you do not intend to
#              actually compute a gradient wrt. said variable "x".
#              Given this info, PyTorch (+ KeOps) is smart enough to
#              skip the computation of unneeded gradients.
p = Variable(torch.randn(  1  ), requires_grad=True ).type(dtype)
g = Variable(torch.randn( M,D ), requires_grad=True ).type(dtype)
x = Variable(torch.randn( N,D ), requires_grad=False).type(dtype)
y = Variable(torch.randn( M,D ), requires_grad=True ).type(dtype)
b = Variable(torch.randn( M,1 ), requires_grad=True ).type(dtype)

# + some random gradient to backprop:
gr = Variable(torch.randn( N,1 ), requires_grad=True ).type(dtype)

a = my_formula(p, g, x, y, b)

# We can compute gradients wrt all Variables - just like with 
# any other PyTorch operator, really.
# Notice the "create_graph=True", which allows us to compute
# higher order derivatives if needed.
[grad_g, grad_y]   = grad( a, [g, y], gr, create_graph=True)

print("(a_i) :", a[:3,:])
print("(∂_p a).g :", grad_g[:3,:] )
print("(∂_y a).g :", grad_y[:3,:])