"""
Writing arbitrary formula with the generic syntax (PyTorch)
===========================================================

This example uses the pyTorch framework.
"""

####################################################################
# It computes the following tensor operation :
#    
# .. math::
#   c_i^u = \sum_j (p-a_j)^2 \exp(x_i^u+y_j^u)
# 
# where 
# 
# * :math:`x`   : a 3000x3 tensor, with entries denoted :math:`x_i^u`
# * :math:`y`   : a 5000x3 tensor, with entries denoted :math:`y_j^u`
# * :math:`a`   : a 5000x1 tensor, with entries denoted :math:`a_j`
# * :math:`p`   : a scalar (entered as a 1x1 tensor)
#
# and the results
#
# * :math:`c`   : a 3000x3 tensor, with entries denoted :math:`c_i^u`
#



####################################################################
# Define our dataset
# ------------------
#
# Standard imports
import time
import torch
from torch.autograd import grad
from pykeops.torch import Genred
import matplotlib.pyplot as plt

#####################################################################
# Declare random inputs

M = 3000
N = 5000

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

type = 'float32' # Could be 'float32' or 'float64'
torchtype = torch.float32 if type == 'float32' else torch.float64

x = torch.randn(M, 3, dtype=torchtype, device=device)
y = torch.randn(N, 3, dtype=torchtype, device=device, requires_grad=True)
a = torch.randn(N, 1, dtype=torchtype, device=device)
p = torch.randn(1, 1, dtype=torchtype, device=device)

####################################################################
# Define the kernel
# -----------------

formula = 'Square(p-a)*Exp(x+y)'
variables = ['x = Vx(3)',  # First arg   : i-variable, of size 3
             'y = Vy(3)',  # Second arg  : j-variable, of size 3
             'a = Vy(1)',  # Third arg   : j-variable, of size 1 (scalar)
             'p = Pm(1)']  # Fourth  arg : Parameter,  of size 1 (scalar)
         
start = time.time()

####################################################################
# The parameter ``reduction_op='Sum'`` together with ``axis=1`` means that the reduction operation is a sum over the second dimension ``j``. Thence the results will be an ``i``-variable.

my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
c = my_routine(x, y, a, p)


####################################################################
# Define the gradient
# -------------------
# Now, let's compute the gradient of :math:`c` with respect to :math:`y`. Note that since :math:`c` is not scalar valued, its "gradient" should be understood as the adjoint of the differential operator, i.e. as the linear operator that takes as input a new tensor :math:`e` with same size as :math:`c` and outputs a tensor :math:`g` with same size as :math:`y` such that for all variation :math:`\delta y` of :math:`y` we have:
#
# .. math::
#
#    \langle dc \cdot \delta y , e \rangle  =  \langle g , \delta y \rangle  =  \langle \delta y , dc^* \cdot e \rangle
#

# Declare a new variable of size Mx3 used as input of the gradient
e = torch.rand_like(c)

# Call the gradient op:
start = time.time()

# PyTorch remark : grad(c, y, e) alone outputs a length 1 tuple, hence the need for [0] at the end.
g = grad(c, y, e)[0]

print('Time to compute gradient of convolution operation with KeOps: ', round(time.time()-start,5), 's')

####################################################################
# the equivalent code with a "vanilla" pytorch implementation

g_torch = ((p - a.transpose(0, 1))[:, None] **2 * torch.exp(x.transpose(0, 1)[:, :, None] + y.transpose(0, 1)[:, None, :]) * e.transpose(0, 1)[:, :, None] ).sum(dim=1).transpose(0, 1)


# compare the results by plotting some values
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(g.detach().cpu().numpy()[:40,i], '-', label='keops')
    plt.plot(g_torch.detach().cpu().numpy()[:40,i], '--', label='numpy')
    plt.legend(loc='upper center')
plt.show()
