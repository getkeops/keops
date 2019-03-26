"""
=========
Multi GPU 
=========

On multi-device clusters,
let's see how to select the card on which a KeOps
operation will be performed.

"""

###############################################################
# Setup
# -------------
# Standard imports:

import numpy as np
import torch

from pykeops.numpy import Genred

###############################################################
# Define the list of gpu ids to be tested:

# By default we assume that there are two GPUs available with 0 and 1 labels:

gpuids = [0,1] if torch.cuda.device_count() > 1 else [0]


###############################################################
# KeOps Kernel
# -------------
# Define some arbitrary KeOps routine:

formula   =  'Square(p-a) * Exp(x+y)'
variables = ['x = Vi(3)','y = Vj(3)','a = Vj(1)','p = Pm(1)']

type = 'float32'  # May be 'float32' or 'float64'


###############################################################
# Tests with the NumPy API
# ------------------------------

my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)

###############################################################
#  Generate some data, stored on the CPU (host) memory:
#
M = 3000
N = 5000
x = np.random.randn(M,3).astype(type)
y = np.random.randn(N,3).astype(type)
a = np.random.randn(N,1).astype(type)
p = np.random.randn(1,1).astype(type)

#########################################
# Launch our routine on the CPU, for reference:
#

c = my_routine(x, y, a, p, backend='CPU')

#########################################
# And on our GPUs, with copies between 
# the Host and Device memories:
#
for gpuid in gpuids:
    d = my_routine(x, y, a, p, backend='GPU', device_id=gpuid)
    print('Convolution operation (numpy bindings, FromHost mode) on gpu device',gpuid,end=' ')
    print('(relative error:', float(np.abs((c - d) / c).mean()), ')')


###############################################################
# Tests with the PyTorch API                   
# ---------------------------

import torch
from pykeops.torch import Genred
my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)

###########################################
# First, we keep the data on the CPU (host) memory:
#
x = torch.from_numpy(x)
y = torch.from_numpy(y)
a = torch.from_numpy(a)
p = torch.from_numpy(p)
c = torch.from_numpy(c)

for gpuid in gpuids:
    d = my_routine(x, y, a, p, backend='GPU', device_id=gpuid)
    print('Convolution operation (pytorch bindings, FromHost mode) on gpu device',gpuid,end=' ')
    print('(relative error:', float(torch.abs((c - d) / c).mean()), ')')

###########################################
# Second, we load the data on the GPU (device) of our choice
# and let KeOps infer the **device_id** automatically:

for gpuid in gpuids:
    with torch.cuda.device(gpuid):
        # Transfer the data from Host to Device memory.
        # N.B.: The first call to ".cuda()" may take several seconds for each device.
        #       This is a known PyTorch issue.
        p, a, x, y = p.cuda(), a.cuda(), x.cuda(), y.cuda()

        # Call our KeOps routine:
        d = my_routine(x, y, a, p, backend='GPU')
        print('Relative error on gpu {}: {:1.3e}'.format(gpuid, 
                float(torch.abs((c - d.cpu()) / c).mean())) )



