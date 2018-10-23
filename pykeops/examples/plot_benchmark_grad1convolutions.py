"""
Benchmark KeOps vs pytorch on convolution gradients
===================================================
"""


#####################################################################
# Generate our dataset
# --------------------

import numpy as np
import timeit
from matplotlib import pyplot as plt

from pykeops.numpy.utils import grad_np_kernel, chain_rules

# size of the test
M = 2000
N = 2000
D = 3
E = 3

type = 'float32'

# declare numpy variables
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


####################################################################
# Start Benchmarks
# ----------------
#
# Define the parameters

print('Times to compute convolutions of size {}x{}:'.format(M, N), end='\n')

####################################################################
# loop over various cases

kernel_to_test = ['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric']
speed_numpy = {i:np.nan for i in kernel_to_test}
speed_pykeops = {i:np.nan for i in kernel_to_test}
speed_pytorch = {i:np.nan for i in kernel_to_test}
speed_pykeops_specific = {i:np.nan for i in kernel_to_test}

for k in  kernel_to_test:
    print('kernel: ' + k)

    # pure numpy
    gnumpy = chain_rules(a, x, y, grad_np_kernel(x, y, sigma, kernel=k), b)
    speed_numpy[k] = timeit.Timer('gnumpy = chain_rules(a, x, y, grad_np_kernel(x, y, sigma, kernel=k), b)', globals=globals()).timeit(5)/5
    print('Time for numpy:            {:.4f}s'.format(speed_numpy[k]))
    
    
    
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
        speed_pykeops[k] = timeit.Timer("g3 = torch.autograd.grad(torch.dot(ac.view(-1), kernel_product(params, xc, yc, bc, mode='sum').view(-1)), xc, create_graph=False)[0]", globals=globals()).timeit(200)/200
        print('Time for Keops+pytorch:    {:.4f}s'.format(speed_pykeops[k]),end='')
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
        speed_pytorch[k] = timeit.Timer("g3 = torch.autograd.grad(torch.dot(ac.view(-1), kernel_product(params, xc, yc, bc, mode='sum').view(-1)), xc, create_graph=False)[0]", globals=globals()).timeit(200)/200
        print('Time for Pytorch:          {:.4f}s'.format(speed_pytorch[k]),end='')
        print('   (absolute error:       ', np.max(np.abs(g3.data.numpy() - gnumpy)), ')')
    except:
        print('Time for Pytorch:             Not Done')
    
    
    
    # specific cuda tiled implementation (if cuda is available)
    try:
        from pykeops.numpy import RadialKernelGrad1conv
        my_conv = RadialKernelGrad1conv(type)
        g1 = my_conv(a, x, y, b, sigma, kernel=k)
        
        speed_pykeops_specific[k] = timeit.Timer('g1 = my_conv(a, x, y, b, sigma, kernel=k)', globals=globals()).timeit(200)/200
        print('Time for keops specific:   {:.4f}s'.format(speed_pykeops_specific[k]),end="")
        print('   (absolute error:       ', np.max(np.abs (g1 - gnumpy)), ')')
    except:
        print('Time for keops cuda specific: Not Done')



####################################################################
# display results

plt.plot(kernel_to_test, speed_numpy.values(), linestyle='--', marker='s', color='green', markersize=15, label='numpy')
plt.plot(kernel_to_test, speed_pykeops.values(), linestyle='--', marker='*', color='red', markersize=15, label='pykeops')
plt.plot(kernel_to_test, speed_pykeops_specific.values(), linestyle='None', marker='d', color='orange', markersize=15, label='pykeops specific routines')
plt.plot(kernel_to_test, speed_pytorch.values(), linestyle='None', marker='p', color='blue', markersize=15, label='pytorch')

plt.yscale('log')
plt.grid(True)
plt.legend(loc='best')
plt.show()