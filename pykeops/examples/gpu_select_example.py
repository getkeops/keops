"""
This example shows how to select the Gpu device on which a KeOps 
operation will be performed, on systems having several devices.

"""

# This example will run only when at least two Gpus are available.
# By default we assume their ids are 0 and 1, but this can be changed here :
gpu0 = 0
gpu1 = 1

import time
import torch
from torch.autograd import grad


import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

from pykeops.torch import Genred

# import pykeops
# pykeops.verbose = False

#--------------------------------------------------------------#
#                   Define our example,                        #
# we use the same example as in generic_syntax_**.py examples  #
#--------------------------------------------------------------#
M = 300000
N = 500000

type = 'float32' # Could be 'float32' or 'float64'
torchtype = torch.float32 if type == 'float32' else torch.float64

x = torch.randn(M, 3, dtype=torchtype)
y = torch.randn(N, 3, dtype=torchtype, requires_grad=True)
a = torch.randn(N, 1, dtype=torchtype)
p = torch.randn(1, 1, dtype=torchtype)



formula = 'Square(p-a)*Exp(x+y)'
variables = ['x = Vx(3)','y = Vy(3)','a = Vy(1)','p = Pm(1)']
my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)

# start = time.time()
# c = my_routine(x, y, a, p, backend='CPU')
# print('Time to compute the convolution operation on the cpu: ', round(time.time()-start,5), 's')

start = time.time()
c = my_routine(x, y, a, p, backend='GPU', device_id=0)
print('Time to compute convolution operation on gpu on device:',round(time.time()-start,5), 's ')

with torch.cuda.device(1):
    # we transfer data on gpu
    p,a,x,y = p.cuda(), a.cuda(), x.cuda(), y.cuda()
    # then call the operations
    start = time.time()
    c = my_routine(x, y, a, p, backend='GPU')
    print('Time to compute convolution operation on gpu:',round(time.time()-start,5), 's ')


