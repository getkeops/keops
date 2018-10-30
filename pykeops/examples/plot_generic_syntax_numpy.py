"""
Writing arbitrary formula with the generic syntax (NumPy)
=========================================================

This example uses a pure numpy framework. 

"""
####################################################################
# It computes the following tensor operation :
#    
# .. math::
#
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
# and its gradient (see below).


####################################################################
# Define our dataset
# ------------------
#
# Standard imports

import numpy as np
from pykeops.numpy import Genred
import matplotlib.pyplot as plt

#####################################################################
# Declare random inputs

M = 3000
N = 5000

type = 'float32'  # May be 'float32' or 'float64'

x = np.random.randn(M,3).astype(type)
y = np.random.randn(N,3).astype(type)
a = np.random.randn(N,1).astype(type)
p = np.random.randn(1,1).astype(type)


####################################################################
# Define the kernel
# -----------------

formula = 'Square(p-a)*Exp(x+y)'
variables = ['x = Vx(3)',  # First arg   : i-variable, of size 3
             'y = Vy(3)',  # Second arg  : j-variable, of size 3
             'a = Vy(1)',  # Third arg   : j-variable, of size 1 (scalar)
             'p = Pm(1)']  # Fourth  arg : Parameter,  of size 1 (scalar)

####################################################################
# The parameter ``reduction_op='Sum'`` together with ``axis=1`` means that the reduction operation is a sum over the second dimension ``j``. Thence the results will be an ``i``-variable.

my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
c = my_routine(x, y, a, p, backend='auto')

####################################################################
# the equivalent code in numpy
c_np = ((p - a.T)[:,np.newaxis] **2 * np.exp(x.T[:,:,np.newaxis] + y.T[:,np.newaxis,:]) ).sum(2).T

# compare the results by plotting them
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(c[:40,i], '-', label='keops')
    plt.plot(c_np[:40,i], '--', label='numpy')
    plt.legend(loc='upper center')
plt.show()

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
e = np.random.randn(M, 3).astype(type)

####################################################################
# Thankfully, KeOps provides an autodiff engine for formulas. Nevertheless, we need to specify some informations by hand: add the gradient operator around formula: ``Grad(formula , variable_to_differentiate, input_of_the_gradient)``
formula_grad =  'Grad(' + formula + ', y, e)'

# This new formula has an a new variable (namely the input variable e)
variables_grad = variables + ['e = Vx(3)'] # Fifth arg: i-variable, of size 3... Just like "c"!

# Note here that the summation  is with respect to the first component (axis=0) in order to get a 'j'-variable
my_grad = Genred(formula_grad, variables_grad, reduction_op='Sum', axis=0, cuda_type=type)

d = my_grad(x, y, a, p, e)

####################################################################
# to generate the equivalent code in numpy: we have to compute explicitly the adjoint of the differential (a.k.a. the derivative). To do so, let see :math:`c^i_u` as a function of :math:`y_j`:
#
# .. math::
#
#   d_j^u = [(\partial_{y} c^u(y))^* e^u]_j = \sum_{i} (p-a_j)^2 \exp(x_i^u+y_j^u) * e_i^u
#
# and implement the formula:

d_np = ((p - a.T)[:, np.newaxis, :] **2 * np.exp(x.T[:, :, np.newaxis] + y.T[:, np.newaxis, :]) * e.T[:, :, np.newaxis]).sum(1).T

# compare the results by plotting:
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(d[:40,i], '-', label='keops')
    plt.plot(d_np[:40,i], '--', label='numpy')
    plt.legend(loc='upper center')
plt.show()
