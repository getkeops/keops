"""
LSE KeOps reduction using the generic syntax 
============================================

This example uses the pyTorch framework.
"""

####################################################################
# It computes the following tensor operation :
#    
# .. math::
#   c_i^u = \log \left[ \sum_j \exp\left( (p-a_j)^2 \exp(x_i^u+y_j^u) \right) \right]
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

x = torch.rand(M, 1, dtype=torchtype)
y = torch.rand(N, 1, dtype=torchtype, requires_grad=True)
a = torch.rand(N, 1, dtype=torchtype)
p = torch.rand(1, 1, dtype=torchtype)

####################################################################
# Define the kernel
# -----------------

formula = 'Square(p-a)*Exp(x+y)'
variables = ['x = Vx(1)',  # First arg   : i-variable, of size 3
             'y = Vy(1)',  # Second arg  : j-variable, of size 3
             'a = Vy(1)',  # Third arg   : j-variable, of size 1 (scalar)
             'p = Pm(1)']  # Fourth  arg : Parameter,  of size 1 (scalar)
         
start = time.time()

####################################################################
# The parameter ``reduction_op='LogSumExp'`` together with ``axis=1`` means that the reduction operation is a log-sum-exp over the second dimension ``j``. Thence the results will be an ``i``-variable.

my_routine = Genred(formula, variables, reduction_op='LogSumExp', axis=1, cuda_type=type)
tmp = my_routine(x, y, a, p, backend='CPU')
# in fact the log-sum-exp operation in Keops computes pairs (m,s) such that the LSE is m+log(s)
c = tmp[:,0] + torch.log(tmp[:,1])

# N.B.: By specifying backend='CPU', we make sure that the result is computed
#       using a simple C++ for loop.
print('Time to compute the convolution operation on the cpu: ', round(time.time()-start,5), 's', end=' ')

####################################################################
# We compare with Log of Sum of Exp:

my_routine2 = Genred('Exp('+formula+')', variables, reduction_op='Sum', axis=1, cuda_type=type)
c2 = torch.log(my_routine2(x, y, a, p, backend='CPU'))[:,0]
print('(relative error: ',((c2-c).norm()/c.norm()).item(), ')')

####################################################################
# Define the gradient
# -------------------
# Now, let's compute the gradient of :math:`c` with respect to :math:`y`. Note that since :math:`c` is not scalar valued, its "gradient" should be understood as the adjoint of the differential operator, i.e. as the linear operator that takes as input a new tensor :math:`e` with same size as :math:`c` and outputs a tensor :math:`g` with same size as :math:`y` such that for all variation :math:`\delta y` of :math:`y` we have:
#
# .. math::
#
#    \langle dc \cdot \delta y , e \rangle  =  \langle g , \delta y \rangle  =  \langle \delta y , dc^* \cdot e \rangle
#

# Declare a new variable of size Mx1 used as input of the gradient
e = torch.rand_like(c)
# Call the gradient op:
start = time.time()
g = grad(c, y, e)[0]
# PyTorch remark : grad(c, y, e) alone outputs a length 1 tuple, hence the need for [0] at the end.

print('Time to compute gradient of convolution operation on the cpu: ', round(time.time()-start,5), 's', end=' ')

####################################################################
# We compare with gradient of Log of Sum of Exp:

g2 = grad(c2, y, e)[0]
print('(relative error: ',((g2-g).norm()/g.norm()).item(), ')')


####################################################################
# Same operations performed on the Gpu
# ------------------------------------
#
# This will of course only work if you have a Gpu...

if torch.cuda.is_available():
    # first transfer data on gpu
    pc, ac, xc, yc, ec = p.cuda(), a.cuda(), x.cuda(), y.cuda(), e.cuda()
    # then call the operations
    start = time.time()
    c3 = my_routine(xc, yc, ac, pc, backend='GPU')
    c3 = c3[:,0] + torch.log(c3[:,1])
    print('Time to compute convolution operation on gpu:',round(time.time()-start,5), 's ', end='')
    print('(relative error:', float(torch.abs((c2 - c3.cpu()) / c2).mean()), ')')
    start = time.time()
    g3 = grad(c3, yc, ec)[0]
    print('Time to compute gradient of convolution operation on gpu:', round(time.time()-start,5), 's ', end='')
    print('(relative error:', float(torch.abs((g2 - g3.cpu()) / g2).mean()), ')')

