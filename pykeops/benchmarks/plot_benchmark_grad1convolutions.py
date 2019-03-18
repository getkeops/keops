"""
Benchmark KeOps vs pytorch on convolution gradients
===================================================
"""


#####################################################################
# Generate our dataset
# --------------------

import numpy as np
import timeit
import matplotlib
from matplotlib import pyplot as plt

from pykeops.numpy.utils import grad_np_kernel, chain_rules

# size of the test
M = 10000
N = 10000
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

for k in kernel_to_test:
    print('kernel: ' + k)

    # pure numpy
    gnumpy = chain_rules(a, x, y, grad_np_kernel(x, y, sigma, kernel=k), b)
    speed_numpy[k] = timeit.repeat('gnumpy = chain_rules(a, x, y, grad_np_kernel(x, y, sigma, kernel=k), b)', globals=globals(), repeat=5, number=1)
    print('Time for numpy:               {:.4f}s'.format(np.median(speed_numpy[k])))
    
    
    
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
        speed_pykeops[k] =  np.array(timeit.repeat("g3 = torch.autograd.grad(torch.dot(ac.view(-1), kernel_product(params, xc, yc, bc, mode='sum').view(-1)), xc, create_graph=False)[0]", globals=globals(), repeat=100, number=4)) / 4
        print('Time for keops generic:       {:.4f}s'.format(np.median(speed_pykeops[k])), end='')
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
        speed_pytorch[k] =  np.array(timeit.repeat("g3 = torch.autograd.grad(torch.dot(ac.view(-1), kernel_product(params, xc, yc, bc, mode='sum').view(-1)), xc, create_graph=False)[0]", globals=globals(), repeat=100, number=4)) / 4
        print('Time for Pytorch:             {:.4f}s'.format(np.median(speed_pytorch[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g3.data.numpy() - gnumpy)), ')')
    except:
        print('Time for Pytorch:             Not Done')
    
    
    
    # specific cuda tiled implementation (if cuda is available)
    try:
        from pykeops.numpy import RadialKernelGrad1conv
        my_conv = RadialKernelGrad1conv(type)
        g1 = my_conv(a, x, y, b, sigma, kernel=k)
        
        speed_pykeops_specific[k] =  np.array(timeit.repeat('g1 = my_conv(a, x, y, b, sigma, kernel=k)', globals=globals(), repeat=100, number=4))/4
        print('Time for keops cuda specific: {:.4f}s'.format(np.median(speed_pykeops_specific[k])), end='')
        print('   (absolute error:       ', np.max(np.abs (g1 - gnumpy)), ')')
    except:
        print('Time for keops cuda specific: Not Done')


####################################################################
# display results

# plot violin plot
plt.violinplot(list(speed_numpy.values()),
               showmeans=False,
               showmedians=True,
               )
plt.violinplot(list(speed_pytorch.values()),
               showmeans=False,
               showmedians=True,
               )
plt.violinplot(list(speed_pykeops.values()),
               showmeans=False,
               showmedians=True,
               )
plt.violinplot(list(speed_pykeops_specific.values()),
               showmeans=False,
               showmedians=True,
               )

plt.xticks([1, 2, 3, 4], kernel_to_test)
plt.yscale('log')
# plt.ylim((0, .01))

plt.grid(True)
plt.xlabel('kernel type')
plt.ylabel('time in s.')

cmap = plt.get_cmap("tab10")
fake_handles = [matplotlib.patches.Patch(color=cmap(i)) for i in range(4)]

plt.legend(fake_handles, ['numpy', 'pytorch', 'pykeops', 'pykeops specific'], loc='best')

plt.show()
