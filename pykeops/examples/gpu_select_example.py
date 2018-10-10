"""
This example shows how to select the Gpu device on which a KeOps 
operation will be performed, on systems having several devices.

"""

import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

# import pykeops
# pykeops.verbose = False


# Define the list of gpu ids to be tested
# By default we assume we have two Gpus available, labeled 0 and 1
gpuids = [0,1]


#--------------------------------------------------------------#
#                  Tests with numpy bindings                   #
#  we use the same example as in generic_syntax_numpy.py       #
#--------------------------------------------------------------#

import numpy as np
from pykeops.numpy import Genred

# formula :
formula = 'Square(p-a)*Exp(x+y)'
variables = ['x = Vx(3)','y = Vy(3)','a = Vy(1)','p = Pm(1)']
my_routine = Genred(formula, variables, reduction_op='Sum', axis=1)

# We first use KeOps numpy bindings
# arrays are created as regular numpy arrays, hence data is on Cpu (host) memory
M = 3000
N = 5000
x = np.random.randn(M,3)
y = np.random.randn(N,3)
a = np.random.randn(N,1)
p = np.random.randn(1,1)

# call to KeOps on Cpu for reference
c = my_routine(x, y, a, p, backend='CPU')

# Internally data is first copied to the selected device memory, operation is performed on selected device,
# and then output data is copied back to Cpu memory (this is what we call the FromHost mode)
for gpuid in gpuids:
    d = my_routine(x, y, a, p, backend='GPU', device_id=gpuid)
    print('Convolution operation (numpy bindings, FromHost mode) on gpu device ',gpuid,' :',end='')
    print('(relative error:', float(torch.abs((c - d) / c).mean()), ')')



#--------------------------------------------------------------#
#                Tests with pytorch bindings                   #
#--------------------------------------------------------------#

import torch
from pykeops.numpy import Genred
my_routine = Genred(formula, variables, reduction_op='Sum', axis=1)

# First we simply make aliases of numpy variables as pytorch tensors, so data is still
# on Cpu memory, thus we still use FromHost mode
x = torch.from_numpy(x)
y = torch.from_numpy(y)
a = torch.from_numpy(a)
p = torch.from_numpy(p)

for gpuid in gpuids:
    d = my_routine(x, y, a, p, backend='GPU', device_id=gpuid)
    print('Convolution operation (pytorch bindings, FromHost mode) on gpu device ',gpuid,' :',end='')
    print('(relative error:', float(torch.abs((c - d) / c).mean()), ')')

# last tests, still using Pytorch bindings, but this time we will use KeOps directly on arrays
# which are already located on Gpu device memory (FromDevice mode)
for gpuid in gpuids:
    # a simple way to select gpu device with Pytorch is via the torch.cuda.device context
    with torch.cuda.device(gpuid):
        # we transfer data on gpu
        p,a,x,y = p.cuda(), a.cuda(), x.cuda(), y.cuda()
        # then call the operation, so it will be performed on the corresponding gpu device
        d = my_routine(x, y, a, p, backend='GPU')
        print('Convolution operation (pytorch bindings, FromHost mode) on gpu device ',gpuid,' :',end='')
        print('(relative error:', float(torch.abs((c - d.cpu()) / c).mean()), ')')



