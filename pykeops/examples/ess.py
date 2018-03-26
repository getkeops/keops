#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 01:18:17 2018

@author: glaunes
"""



# Add pykeops to the path
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

# Standard imports
import torch
from torch          import Tensor
from torch.autograd import Variable, grad
from pykeops.torch.generic_sum       import GenericSum
from pykeops.torch.generic_logsumexp import GenericLogSumExp

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def my_formula(p, x, y, b, backend = "auto") :
    aliases = [ "P = Pm(0,1)",                         #  a parameter, 1st argument, dim 1. 
                "X = Vx(1," + str(x.shape[1]) + ") ",  # indexed by i, 2nd argument, dim D.
                "Y = Vy(2," + str(y.shape[1]) + ") ",  # indexed by j, 3rd argument, dim D.
                "B = Vy(3," + str(y.shape[1]) + ") ",  # indexed by j, 4th argument, dim D.
              ]
    formula = "Exp( -P*SqDist(X,Y) ) * B"

    signature   =   [ (D, 0), (1, 2), (D, 0), (D, 1), (D, 1) ]

    sum_index   = 0 # the result is indexed by "i"; for "j", use "1"

    genconv = GenericSum.apply
    a  = genconv( backend, aliases, formula, signature, sum_index, p, x, y, b)
    return a


# Test ========================================================================
# Define our dataset
N = 1000 ; M = 2000 ; D = 3
p = Variable(torch.randn(  1  ), requires_grad=True ).type(dtype)
x = Variable(torch.randn( N,D ), requires_grad=False).type(dtype)
y = Variable(torch.randn( M,D ), requires_grad=True ).type(dtype)
b = Variable(torch.randn( M,D ), requires_grad=True ).type(dtype)

g = Variable(torch.randn( N,D ), requires_grad=True ).type(dtype)

for backend in ["auto"] :
    print("Backend :", backend, "============================" )
    a = my_formula(p, x, y, b, backend=backend)

    [grad_p, grad_y]   = grad( a, [p, y], g)

    print("(a_i) :", a[:3,:])
    print("(∂_p a).g :", grad_p )
    print("(∂_y a).g :", grad_y[:3,:])