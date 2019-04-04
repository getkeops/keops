"""
Radial kernels convolutions
===========================

This benchmark compares the performances of KeOps versus Numpy and PyTorch on various radial
kernels convolutions. Namely it computes:

.. math::

   a_i = \sum_{j=1}^N f\Big(\\frac{\|x_i-y_j\|}{\sigma}\Big) b_j, \quad \\text{ for all } i=1,\cdots,M

where :math:`f` is a Gauss or Cauchy or Laplace or inverse multiquadric kernel. See e.g. `wikipedia  <https://en.wikipedia.org/wiki/Radial_basis_function>`_

 
"""

#####################################################################
# Setup
# -----
# Standard imports:

import numpy as np
import timeit
import matplotlib
from matplotlib import pyplot as plt
from pykeops.numpy.utils import np_kernel

######################################################################
# Benchmark specifications:
#

M = 10000  # Number of points x_i
N = 10000  # Number of points y_j
D = 3      # Dimension of the x_i's and y_j's
E = 3      # Dimension of the b_j's
REPEAT = 10  # Number of loops per test

dtype = 'float32'

######################################################################
# Create some random input data:
#

x = np.random.randn(M, D).astype(dtype)  # Target points
y = np.random.randn(N, D).astype(dtype)  # Source points
b = np.random.randn(N, E).astype(dtype)  # Source signal
sigma = np.array([2.4]).astype(dtype)    # Kernel radius

######################################################################
# And load it as PyTorch variables:
#

try:
    import torch
    
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    torchtype = torch.float32 if dtype == 'float32' else torch.float64

    xc = torch.tensor(x, dtype=torchtype, device=device)
    yc = torch.tensor(y, dtype=torchtype, device=device)
    bc = torch.tensor(b, dtype=torchtype, device=device)
    sigmac = torch.tensor(sigma, dtype=torchtype, device=device)

except:
    pass


####################################################################
# Convolution Benchmarks
# ----------------------
# 
# We loop over four different kernels:
#

kernel_to_test = ['gaussian', 'laplacian', 'cauchy', 'inverse_multiquadric']

#####################################################################
# With four backends: Numpy, vanilla PyTorch, Generic KeOps reductions
# and a specific, handmade legacy CUDA code for kernel convolutions:
#

speed_numpy = {i:np.nan for i in kernel_to_test}
speed_pykeops = {i:np.nan for i in kernel_to_test}
speed_pytorch = {i:np.nan for i in kernel_to_test}
speed_pykeops_specific = {i:np.nan for i in kernel_to_test}

print('Timings for {}x{} convolutions:'.format(M, N))

for k in kernel_to_test:
    print('kernel: ' + k)
    
    # Pure numpy
    g_numpy =  np.matmul( np_kernel(x, y, sigma, kernel=k ), b)
    speed_numpy[k] = timeit.repeat('gnumpy = np.matmul( np_kernel(x, y, sigma, kernel=k), b)', 
                                    globals=globals(), repeat=5, number=1)
    print('Time for NumPy:               {:.4f}s'.format( np.median(speed_numpy[k])) )



    # Vanilla pytorch (with cuda if available, and cpu otherwise)
    try:
        from pykeops.torch import Kernel, kernel_product
        
        params = {
            'id': Kernel(k + '(x,y)'),
            'gamma': 1. / (sigmac**2),
            'backend': 'pytorch',
        }
        
        g_pytorch = kernel_product(params, xc, yc, bc, mode='sum').cpu()
        torch.cuda.synchronize()
        speed_pytorch[k] = np.array(timeit.repeat(
            "kernel_product(params, xc, yc, bc, mode='sum'); torch.cuda.synchronize()", 
            globals=globals(), repeat=REPEAT, number=4)) / 4

        print('Time for PyTorch:             {:.4f}s'.format(np.median(speed_pytorch[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g_pytorch.numpy() - g_numpy)),')')
    except:
        print('Time for PyTorch:             Not Done')
    
    
    
    # Keops: generic tiled implementation (with cuda if available, and cpu otherwise)
    try:
        from pykeops.torch import Kernel, kernel_product
    
        params = {
            'id': Kernel(k + '(x,y)'),
            'gamma': 1. / (sigmac**2),
            'backend': 'auto',
        }

        g_keops = kernel_product(params, xc, yc, bc,  mode='sum').cpu()
        torch.cuda.synchronize()
        speed_pykeops[k] = np.array(timeit.repeat(
            "kernel_product(params, xc, yc, bc, mode='sum'); torch.cuda.synchronize()", 
            globals=globals(), repeat=REPEAT, number=4)) / 4
        print('Time for KeOps generic:       {:.4f}s'.format(np.median(speed_pykeops[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g_keops.data.numpy() - g_numpy)), ')')
    except:
        print('Time for KeOps generic:       Not Done')
    
    
    
    # Specific cuda tiled implementation (if cuda is available)
    try:
        from pykeops.numpy import RadialKernelConv
        my_conv = RadialKernelConv(dtype)
        g_specific = my_conv(x, y, b, sigma, kernel=k)
        torch.cuda.synchronize()
        speed_pykeops_specific[k] = np.array(timeit.repeat(
            "my_conv(x, y, b, sigma, kernel=k); torch.cuda.synchronize()", 
            globals=globals(), repeat=REPEAT, number=4)) / 4
        print('Time for KeOps cuda specific: {:.4f}s'.format(np.median(speed_pykeops_specific[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g_specific - g_numpy)),')')
    except:
        print('Time for KeOps cuda specific: Not Done')


####################################################################
# Display results
# ---------------
#

cmap = plt.get_cmap("tab10")

def draw_plot(data, edge_color, fill_color, pos):
    bp = ax.boxplot(data, patch_artist=True, positions=pos)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)  

fig, ax = plt.subplots()
draw_plot(list(speed_numpy.values()), cmap(0), cmap(0), [1,7,13,19])
draw_plot(list(speed_pytorch.values()), cmap(1), cmap(1), [2,8,14,20])
draw_plot(list(speed_pykeops.values()), cmap(2), cmap(2), [3,9,15,21])
draw_plot(list(speed_pykeops_specific.values()), cmap(3), cmap(3), [4,10,16,22])

plt.xticks([0, 2.5, 8.5, 14.5, 20.5, 23], [""] + kernel_to_test + [""])
plt.yscale('log')

plt.grid(True)
plt.xlabel('kernel type')
plt.ylabel('time in s.')

fake_handles = [matplotlib.patches.Patch(color=cmap(i)) for i in range(4)]

plt.legend(fake_handles, ['NumPy', 'PyTorch', 'KeOps', 'KeOps specific'], loc='upper center')

plt.show()
