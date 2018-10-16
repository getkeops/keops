"""
Example of KeOps reduction using the generic syntax.
====================================================

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

#####################################################################
# Declare random inputs

M = 3000
N = 5000

type = 'float32' # Could be 'float32' or 'float64'
torchtype = torch.float32 if type == 'float32' else torch.float64

x = torch.randn(M, 3, dtype=torchtype)
y = torch.randn(N, 3, dtype=torchtype, requires_grad=True)
a = torch.randn(N, 1, dtype=torchtype)
p = torch.randn(1, 1, dtype=torchtype)

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
c = my_routine(x, y, a, p, backend='CPU')

# N.B.: By specifying backend='CPU', we make sure that the result is computed
#       using a simple C++ for loop.

print('Time to compute the convolution operation on the cpu: ', round(time.time()-start,5), 's')

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
g = grad(c, y, e)[0]
# PyTorch remark : grad(c, y, e) alone outputs a length 1 tuple, hence the need for [0] at the end.

print('Time to compute gradient of convolution operation on the cpu: ', round(time.time()-start,5), 's')



####################################################################
# Same operations performed on the Gpu
# ------------------------------------
#
# This will of course only work if you have a Gpu...


if torch.cuda.is_available():
    # first transfer data on gpu
    p,a,x,y,e = p.cuda(), a.cuda(), x.cuda(), y.cuda(), e.cuda()
    # then call the operations
    start = time.time()
    c2 = my_routine(x, y, a, p, backend='GPU')
    print('Time to compute convolution operation on gpu:',round(time.time()-start,5), 's ', end='')
    print('(relative error:', float(torch.abs((c - c2.cpu()) / c).mean()), ')')
    start = time.time()
    g2 = grad(c2, y, e)[0]
    print('Time to compute gradient of convolution operation on gpu:', round(time.time()-start,5), 's ', end='')
    print('(relative error:', float(torch.abs((g - g2.cpu()) / g).mean()), ')')
