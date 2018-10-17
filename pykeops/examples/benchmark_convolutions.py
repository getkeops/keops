"""
Benchmark KeOps vs pytorch on simple convolutions
=================================================
"""

#####################################################################
# Generate our dataset
# --------------------

import numpy as np
import time, timeit

from pykeops.numpy.utils import np_kernel

# size of the test
M = 2000
N = 300
D = 3
E = 3

type = 'float32'

# declare numpy variables 
x = np.random.randn(M, D).astype(type)
y = np.random.randn(N, D).astype(type)
b = np.random.randn(N, E).astype(type)
sigma = np.array([2.4]).astype(type)

# declare their torch counterparts
try:
    import torch

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    torchtype = torch.float32 if type == 'float32' else torch.float64

    xc = torch.tensor(x, dtype=torchtype, device=device)
    yc = torch.tensor(y, dtype=torchtype, device=device)
    bc = torch.tensor(b, dtype=torchtype, device=device)
    sigmac = torch.tensor(sigma, dtype=torchtype, device=device)

except:
    pass


####################################################################
# Start Benchmarks
# ----------------
# 
# Define the parameters

enable_GC = False # Garbage collection?
GC = 'gc.enable();' if enable_GC else 'pass;'
LOOPS = 200
print('Times to compute ', LOOPS, ' convolutions of size {}x{}:'.format(M, N), end='\n')

####################################################################
# loop over various cases

for k in (['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric']):
    print('kernel: ' + k)
    
    # pure numpy
    gnumpy =  np.matmul(np_kernel(x,y,sigma,kernel=k),b)
    speed_numpy = timeit.Timer('gnumpy = np.matmul(np_kernel(x,y,sigma,kernel=k),b)',
                               GC, globals=globals(),
                               timer=time.time).timeit(LOOPS)
    print('Time for Python:              {:.4f}s'.format(speed_numpy))

    # keops + pytorch : generic tiled implementation (with cuda if available else uses cpu)
    try:
        from pykeops.torch import Kernel, kernel_product

        params = {
            'id': Kernel(k+'(x,y)'),
            'gamma': 1. / (sigmac * sigmac),
            'backend': 'auto',
        }
        g1 = kernel_product(params, xc, yc, bc,  mode='sum').cpu()
        speed_pykeops_gen = timeit.Timer("g1 = kernel_product(params, xc, yc, bc, mode='sum').cpu()",
                                         GC, globals=globals(),
                                         timer=time.time).timeit(LOOPS)
        print('Time for keops generic:       {:.4f}s'.format(speed_pykeops_gen),end='')
        print('   (absolute error:       ', np.max(np.abs(g1.data.numpy() - gnumpy)), ')')
    except:
        print('Time for keops generic:       Not Done')

    # vanilla pytorch (with cuda if available else uses cpu)
    try:
        from pykeops.torch import Kernel, kernel_product
    
        params = {
            'id': Kernel(k + '(x,y)'),
            'gamma': 1. / (sigmac * sigmac),
            'backend': 'pytorch',
        }
        
        g0 = kernel_product(params, xc, yc, bc, mode='sum')
        speed_pytorch = timeit.Timer("g0 = kernel_product(params, xc, yc, bc, mode='sum')",
                                     GC, globals=globals(),
                                     timer=time.time).timeit(LOOPS)
        print('Time for Pytorch:             {:.4f}s'.format(speed_pytorch),end='')
        print('   (absolute error:       ', np.max(np.abs(g0.cpu().numpy() - gnumpy)),')')
    except:
        print('Time for Pytorch:             Not Done')

    # specific cuda tiled implementation (if cuda is available)
    try:
        from pykeops.numpy import RadialKernelConv
        my_conv = RadialKernelConv(type)
        g2 = my_conv(x, y, b, sigma, kernel=k)
        speed_pykeops = timeit.Timer('g2 = my_conv(x, y, b, sigma, kernel=k)',
                                     GC, globals=globals(),
                                     timer=time.time).timeit(LOOPS)
        print('Time for keops cuda specific: {:.4f}s'.format(speed_pykeops), end='')
        print('   (absolute error:       ', np.max(np.abs(g2 - gnumpy)),')')
    except:
        print('Time for keops cuda specific: Not Done')
