"""
=========
Multi GPU
=========

This example shows how to select the Gpu device on which a KeOps
operation will be performed, on systems having several devices.

"""

import numpy as np
import torch

from pykeops.numpy import Genred

###############################################################
# GPU selection
# -------------
#
# Define the list of gpu ids to be tested. By default we assume we have two Gpus available, labeled 0 and 1

gpuids = [0,1] if torch.cuda.device_count() > 1 else [0]


###############################################################
# Kernel
# -------------
# Formula :

formula = 'Square(p-a)*Exp(x+y)'
variables = ['x = Vx(3)','y = Vy(3)','a = Vy(1)','p = Pm(1)']

type = 'float32'  # May be 'float32' or 'float64'

###############################################################
#  Tests with numpy bindings
#  -------------------------
#  we use the same example as in generic_syntax_numpy.py
#

my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)

# We first use KeOps numpy bindings
# arrays are created as regular numpy arrays, hence data is on Cpu (host) memory
M = 3000
N = 5000
x = np.random.randn(M,3).astype(type)
y = np.random.randn(N,3).astype(type)
a = np.random.randn(N,1).astype(type)
p = np.random.randn(1,1).astype(type)

# call to KeOps on Cpu for reference
c = my_routine(x, y, a, p, backend='CPU')

# Internally data is first copied to the selected device memory,
# operation is performed on selected device, and then output data
# is copied back to Cpu memory (this is what we call the FromHost mode)
for gpuid in gpuids:
    d = my_routine(x, y, a, p, backend='GPU', device_id=gpuid)
    print('Convolution operation (numpy bindings, FromHost mode) on gpu device',gpuid,end=' ')
    print('(relative error:', float(np.abs((c - d) / c).mean()), ')')