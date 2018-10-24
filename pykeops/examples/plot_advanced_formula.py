"""
Using advanced syntax in formula
================================
"""

####################################################################
# In this demo, we show how to write formulas using advanced syntax with KeOps.
#
#
# In this demo file, given:
#
# - :math:`p`,   a vector of size 2
# - :math:`x_i`, an N-by-D array
# - :math:`y_j`, an M-by-D array
#
# We will compute :math:`(a_i)`, an N-by-D array given by:
#
# .. math::
#
#   a_i = \sum_{j=1}^M (\langle x_i,y_j \rangle^2) (p_0 x_i + p_1 y_j) 
# 
# N.B.: if you are just interested in writing a new 'kernel' formula, you may use the (more convenient) syntax showcased in :doc:`custom_kernel.py <../_auto_examples/plot_generic_syntax_pytorch>`.


####################################################################
# Standard imports

import torch
from torch.autograd import grad
from pykeops.torch import Genred

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def my_formula(p, x, y, backend = 'auto') :
    """
    Applies a custom formula on the torch variables P, X and Y.
    Two backends are provided, so that we can check the correctness
    of both implementations.
    """
    # Vanilla PyTorch implementation
    if backend == 'pytorch':
        scals = (torch.mm(x, y.t()))**2 # Memory-intensive computation!
        a = p[0] * scals.sum(dim=1).view(-1,1) * x + p[1] * (torch.mm(scals, y))
        return a
    
    # KeOps implementation
    else:
        # We now expose the low-level syntax of KeOps.
        # The library relies on vector 'Variables' which can be either:
        # - indexed by 'i' ('x' variables, category 0)
        # - indexed by 'j' ('y' variables, category 1)
        # - constant across the reduction ('parameters', category 2)
        #
        # First of all, we must define a 'who's who' list of the variables used,
        # by specifying their categories, index in the arguments' list, and dimensions:
        variables = ['P = Pm(2)',                        # 1st argument,  a parameter, dim 2.
                     'X = Vx(' + str(x.shape[1]) + ')',  # 2nd argument, indexed by i, dim D.
                     'Y = Vy(' + str(y.shape[1]) + ')']  # 3rd argument, indexed by j, dim D.

        # The actual formula:
        # a_i   =   (<x_i,y_j>**2) * (       p[0]*x_i  +       p[1]*y_j )
        formula = 'Pow( (X|Y) , 2) * ( (Elem(P,0) * X) + (Elem(P,1) * Y) )'

        my_routine = Genred(formula, variables, reduction_op='Sum', axis=1)
        a = my_routine(p, x, y, backend=backend)
        return a

####################################################################
# Define our dataset
# ------------------

N = 1000
M = 2000
D = 3

# PyTorch tip: do not 'require_grad' of 'x' if you do not intend to
#              actually compute a gradient wrt. said variable 'x'.
#              Given this info, PyTorch (+ KeOps) is smart enough to
#              skip the computation of unneeded gradients.
p = torch.randn(2,    requires_grad=True , device=device)
x = torch.randn(N, D, requires_grad=False, device=device)
y = torch.randn(M, D, requires_grad=True , device=device)

# + some random gradient to backprop:
g = torch.randn(N, D, requires_grad=True, device=device)

####################################################################
# Perform the tests
# -----------------

for backend in ['pytorch', 'auto'] :
    print('Backend :', backend, '============================' )
    a = my_formula(p, x, y, backend=backend)

    # We can compute gradients wrt all Variables - just like with 
    # any other PyTorch operator.
    # Notice the 'create_graph=True', which allows us to compute
    # higher order derivatives if needed.
    [grad_p, grad_y] = grad(a, [p, y], g, create_graph=True)

    print('(a_i) :', a[:3,:])
    print('(∂_p a).g :', grad_p )
    print('(∂_y a).g :', grad_y[:3,:])
