#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:29:30 2018

@author: glaunes
"""

import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import torch
from torch.autograd import Variable

import time

from pykeops.torch.kernels import Kernel, kernel_product

use_cuda = torch.cuda.is_available()

###### main settings for this example ############################
deviceId = 0             # id of Gpu device (in case Gpu is  used)
##################################################################

backend_keops = "auto"

# function to transfer data on Gpu only if we use the Gpu
def CpuOrGpu(x):
    if use_cuda:
        if type(x)==tuple:
            x = tuple(map(lambda x:x.cuda(),x))
        else:
            x = x.cuda()
    return x


# define Gaussian kernel (K(x,y)b)_i = sum_j exp(-|xi-yj|^2)bj
def GaussKernel(sigma,lib="keops"):
    if lib=="pytorch":
        oos2 = 1/sigma**2
        def K(x,y,b):
            return torch.exp(-oos2*torch.sum((x[:,None,:]-y[None,:,:])**2,dim=2))@b
        return K
    elif lib=="keops":
        def K(x,y,b):
            params = {
                "id"      : Kernel("gaussian(x,y)"),
                "gamma"   : 1/sigma**2,
                "backend" : backend_keops
            }
            return kernel_product( x,y,b, params)
        return K

def RunExample(kernel_lib="keops"):
    m, n, dp, dv = 2000, 300, 3, 3
    x = Variable(CpuOrGpu(torch.randn(m,dp)), requires_grad=True)
    y = Variable(CpuOrGpu(torch.randn(n,dp)), requires_grad=True)
    b = Variable(CpuOrGpu(torch.randn(n,dv)), requires_grad=True)
    sigma = Variable(CpuOrGpu(torch.FloatTensor([1.5])))
    K = GaussKernel(sigma,kernel_lib)
    
    start = time.time()
    for i in range(200):
        a=K(x,y,b)
    print('time with '+kernel_lib+' : ',round(time.time()-start,2),' seconds')
    
# run the example
if use_cuda:
    with torch.cuda.device(deviceId):
        for k in range(3):
            RunExample("keops")
        for k in range(3):
            RunExample("pytorch")
        for k in range(3):
            RunExample("keops")
        



