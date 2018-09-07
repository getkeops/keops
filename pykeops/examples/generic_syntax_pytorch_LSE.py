"""
Example of LSE KeOps reduction using the generic syntax. This example
uses the pyTorch framework.

# this example computes the following tensor operation :
# inputs :
#   x   : a 3000x1 tensor, with entries denoted x_i^u
#   y   : a 5000x1 tensor, with entries denoted y_j^u
#   a   : a 5000x1 tensor, with entries denoted a_j
#   p   : a scalar (entered as a 1x1 tensor)
# output :
#   c   : a 3000x3 tensor, with entries denoted c_i^u, such that
#   c_i^u = sum_j (p-a_j)^2 exp(x_i^u+y_j^u)

"""

#--------------------------------------------------------------#
#                     Standard imports                         #
#--------------------------------------------------------------#
import time
import torch
from torch.autograd import grad


#--------------------------------------------------------------#
#   Please use the "verbose" compilation mode for debugging    #
#--------------------------------------------------------------#
import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

from pykeops.torch import Genred

# import pykeops
# pykeops.verbose = False

#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#
M = 3000
N = 5000

type = 'float32' # Could be 'float32' or 'float64'
torchtype = torch.float32 if type == 'float32' else torch.float64

x = torch.rand(M, 1, dtype=torchtype)
y = torch.rand(N, 1, dtype=torchtype, requires_grad=True)
a = torch.rand(N, 1, dtype=torchtype)
p = torch.rand(1, 1, dtype=torchtype)

#--------------------------------------------------------------#
#                        Kernel                                #
#--------------------------------------------------------------#
formula = 'Square(p-a)*Exp(x+y)'
variables = ['x = Vx(1)',  # First arg   : i-variable, of size 3
             'y = Vy(1)',  # Second arg  : j-variable, of size 3
             'a = Vy(1)',  # Third arg   : j-variable, of size 1 (scalar)
             'p = Pm(1)']  # Fourth  arg : Parameter,  of size 1 (scalar)
         
start = time.time()

# The parameter reduction_op='LogSumExp' together with axis=1 means that the reduction operation
# is a log-sum-exp over the second dimension j. Thence the results will be an i-variable.
my_routine = Genred(formula, variables, reduction_op='LogSumExp', axis=1, cuda_type=type)
tmp = my_routine(x, y, a, p, backend='CPU')
# in fact the log-sum-exp operation in Keops computes pairs (m,s) such that the LSE is m+log(s)
c = tmp[:,0] + torch.log(tmp[:,1])

# N.B.: By specifying backend='CPU', we make sure that the result is computed
#       using a simple C++ for loop.
print('Time to compute the convolution operation on the cpu: ', round(time.time()-start,5), 's', end=' ')


# We compare with Log of Sum of Exp :
my_routine2 = Genred('Exp('+formula+')', variables, reduction_op='Sum', axis=1, cuda_type=type)
c2 = torch.log(my_routine2(x, y, a, p, backend='CPU'))[:,0]
print('(relative error: ',((c2-c).norm()/c.norm()).item(), ')')

#--------------------------------------------------------------#
#                        Gradient                              #
#--------------------------------------------------------------#
# Now, let's compute the gradient of "c" with respect to y. 
# Note that since "c" is not scalar valued, its "gradient" should be understood as 
# the adjoint of the differential operator, i.e. as the linear operator that takes as input 
# a new tensor "e" with same size as "c" and outputs a tensor "g" with same size as "y"
# such that for all variation δy of y :
#    < dc.δy , e >_2  =  < g , δy >_2  =  < δy , dc*.e >_2

# New variable of size Mx1 used as input of the gradient
e = torch.rand_like(c)
# Call the gradient op:
start = time.time()
g = grad(c, y, e)[0]
# PyTorch remark : grad(c, y, e) alone outputs a length 1 tuple, hence the need for [0] at the end.

print('Time to compute gradient of convolution operation on the cpu: ', round(time.time()-start,5), 's', end=' ')

# We compare with gradient of Log of Sum of Exp :
g2 = grad(c2, y, e)[0]
print('(relative error: ',((g2-g).norm()/g.norm()).item(), ')')


#--------------------------------------------------------------#
#            same operations performed on the Gpu              #
#--------------------------------------------------------------#
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

