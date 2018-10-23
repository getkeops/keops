"""
Benchmark KeOps vs pytorch on simple convolutions
=================================================
"""

#####################################################################
# Generate our dataset
# --------------------

import numpy as np
import timeit
import matplotlib
from matplotlib import pyplot as plt


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
    gnumpy =  np.matmul(np_kernel(x,y,sigma,kernel=k),b)
    speed_numpy[k] = timeit.repeat('gnumpy = np.matmul(np_kernel(x,y,sigma,kernel=k),b)', globals=globals(),repeat=5, number=1)
    print('Time for Python:              {:.4f}s'.format(np.median(speed_numpy[k])))



    # keops + pytorch : generic tiled implementation (with cuda if available else uses cpu)
    try:
        from pykeops.torch import Kernel, kernel_product
    
        params = {
            'id': Kernel(k+'(x,y)'),
            'gamma': 1. / (sigmac * sigmac),
            'backend': 'auto',
        }
        g1 = kernel_product(params, xc, yc, bc,  mode='sum').cpu()
        speed_pykeops[k] = np.array(timeit.repeat("g1 = kernel_product(params, xc, yc, bc, mode='sum')", globals=globals(), repeat=50, number=4)) / 4
        print('Time for keops generic:       {:.4f}s'.format(np.median(speed_pykeops[k])), end='')
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
        speed_pytorch[k] = np.array(timeit.repeat("g0 = kernel_product(params, xc, yc, bc, mode='sum')", globals=globals(), repeat=50, number=4)) / 4
        print('Time for Pytorch:             {:.4f}s'.format(np.median(speed_pytorch[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g0.cpu().numpy() - gnumpy)),')')
    except:
        print('Time for Pytorch:             Not Done')



    # specific cuda tiled implementation (if cuda is available)
    try:
        from pykeops.numpy import RadialKernelConv
        my_conv = RadialKernelConv(type)
        g2 = my_conv(x, y, b, sigma, kernel=k)
        speed_pykeops_specific[k] = np.array(timeit.repeat('g2 = my_conv(x, y, b, sigma, kernel=k)', globals=globals(), repeat=50, number=4)) / 4
        print('Time for keops cuda specific: {:.4f}s'.format(np.median(speed_pykeops_specific[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g2 - gnumpy)),')')
    except:
        print('Time for keops cuda specific: Not Done')


####################################################################
# display results

#axes = plt.axis()

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
plt.ylim((0, .01))

plt.grid(True)
plt.xlabel('kernel type')
plt.ylabel('time in s.')

cmap = plt.get_cmap("tab10")
fake_handles = [matplotlib.patches.Patch(color=cmap(i)) for i in range(4)]

plt.legend(fake_handles, ['numpy', 'pytorch', 'pykeops', 'pykeos specific'], loc=9, bbox_to_anchor=(1.3, 0.5))

plt.show()
