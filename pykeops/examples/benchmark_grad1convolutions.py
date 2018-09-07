import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import numpy as np
import time, timeit

from pykeops.numpy.utils import differences, squared_distances,grad_np_kernel, chain_rules

N = 1500
M = 300
D = 3
E = 3

type = 'float32'

# declare numpy 

a = np.random.rand(N, E).astype(type)
x = np.random.rand(N, D).astype(type)
y = np.random.rand(M, D).astype(type)
b = np.random.rand(M, E).astype(type)
sigma = np.array([0.4]).astype(type)

# declare the torch counterpart
try:
    import torch
    
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    torchtype = torch.float32 if type == 'float32' else torch.float64

    ac = torch.tensor(a, dtype=torchtype, device=device)
    xc = torch.tensor(x, dtype=torchtype, device=device, requires_grad=True)
    yc = torch.tensor(y, dtype=torchtype, device=device)
    bc = torch.tensor(b, dtype=torchtype, device=device)
    sigmac = torch.tensor(sigma, dtype=torchtype, device=device)

except:
    pass

############################################################
#                       Benchmark
############################################################

enable_GC = False # Garbage collection?
GC = 'gc.enable();' if enable_GC else 'pass;'
LOOPS = 200
print('Times to compute ', LOOPS, ' grad-convolutions of size {}x{}:'.format(M, N), end='\n')


for k in (['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric']):
    print('----------------------------------- kernel: ' + k)

    # pure numpy
    gnumpy = chain_rules(a, x, y, grad_np_kernel(x, y, sigma, kernel=k), b)
    speed_numpy = timeit.Timer('gnumpy = chain_rules(a, x, y, grad_np_kernel(x, y, sigma, kernel=k), b)',
                               GC, globals = globals(),
                               timer = time.time).timeit(LOOPS)
    print('Time for numpy:            {:.4f}s'.format(speed_numpy))

    # keops + pytorch : generic tiled implementation (with cuda if available else uses cpu)
    try:
        from pykeops.torch import Kernel, kernel_product
    
        params = {
            'id': Kernel(k + '(x,y)'),
            'gamma': 1. / (sigmac * sigmac),
            'backend': 'auto',
        }
    
        aKxy_b = torch.dot(ac.view(-1), kernel_product(params, xc, yc, bc, mode='sum').view(-1))
        g3 = torch.autograd.grad(aKxy_b, xc, create_graph=False)[0].cpu()
        speed_keops = timeit.Timer("g3 = torch.autograd.grad(torch.dot(ac.view(-1), kernel_product(params, xc, yc, bc, mode='sum').view(-1)), xc, create_graph=False)[0]",
                                   GC, globals = globals(),
                                   timer = time.time).timeit(LOOPS)
        print('Time for Keops+pytorch:    {:.4f}s'.format(speed_keops),end='')
        print('   (absolute error:       ', np.max(np.abs(g3.data.numpy() - gnumpy)), ')')
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

        aKxy_b = torch.dot(ac.view(-1), kernel_product(params, xc, yc, bc, mode='sum').view(-1))
        g3 = torch.autograd.grad(aKxy_b, xc, create_graph=False)[0].cpu()
        speed_keops = timeit.Timer("g3 = torch.autograd.grad(torch.dot(ac.view(-1), kernel_product(params, xc, yc, bc, mode='sum').view(-1)), xc, create_graph=False)[0]",
                                   GC, globals = globals(),
                                   timer = time.time).timeit(LOOPS)
        print('Time for Pytorch:          {:.4f}s'.format(speed_keops),end='')
        print('   (absolute error:       ', np.max(np.abs(g3.data.numpy() - gnumpy)), ')')
    except:
        print('Time for Pytorch:             Not Done')
        
    # specific cuda tiled implementation (if cuda is available)
    try:
        from pykeops.numpy import RadialKernelGrad1conv
        my_conv = RadialKernelGrad1conv(type)
        g1 = my_conv(a, x, y, b, sigma, kernel=k)
        
        speed_pykeops = timeit.Timer('g1 = my_conv(a, x, y, b, sigma, kernel=k)',
                                     GC, globals = globals(),
                                     timer = time.time).timeit(LOOPS)
        print('Time for keops specific:   {:.4f}s'.format(speed_pykeops),end="")
        print('   (absolute error:       ', np.max(np.abs (g1 - gnumpy)), ')')
    except:
        print('Time for keops cuda specific: Not Done')

