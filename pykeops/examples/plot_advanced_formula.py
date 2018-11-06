"""
Using advanced syntax in formulas
=================================

In this demo, we show how to write generic formulas using KeOps syntax.
"""

####################################################################
# Standard imports

import torch
from pykeops.torch import Genred
import matplotlib.pyplot as plt

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################################################################
# Define our dataset
# ------------------
#
# In this demo file, given:
#
# - :math:`p`,   a vector of size 2
# - :math:`x_i`, an N-by-D array
# - :math:`y_j`, an M-by-D array

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
# Accessing a coordinate of a parameter
# -------------------------------------
#
# We will compute :math:`(a_i)`, an N-by-D array given by:
#
# .. math::
#
#   a_i = \sum_{j=1}^M (\langle x_i,y_j \rangle^2) (p_0 x_i + p_1 y_j)
#
# where the two real parameters are stored in a 2-vector :math:`p=(p_0,p_1)`

# Keops implementation
formula = 'Pow((X|Y), 2) * ((Elem(P, 0) * X) + (Elem(P, 1) * Y))'
variables = ['P = Pm(2)',                        # 1st argument,  a parameter, dim 2.
             'X = Vx(3)',  # 2nd argument, indexed by i, dim D.
             'Y = Vy(3)']  # 3rd argument, indexed by j, dim D.

my_routine = Genred(formula, variables, reduction_op='Sum', axis=1)
a_keops = my_routine(p, x, y)

# Vanilla PyTorch implementation
scals = (torch.mm(x, y.t())) ** 2  # Memory-intensive computation!
a_pytorch = p[0] * scals.sum(1).view(-1, 1) * x + p[1] * (torch.mm(scals, y))

# Check the results
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(a_keops.detach().cpu().numpy()[:40, i], '-', label='keops')
    plt.plot(a_pytorch.detach().cpu().numpy()[:40, i], '--', label='numpy')
    plt.legend(loc='upper center')
plt.show()
