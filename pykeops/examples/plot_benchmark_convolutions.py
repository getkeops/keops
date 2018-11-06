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
M = 10000
N = 10000
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
    print('Time for numpy:               {:.4f}s'.format(np.median(speed_numpy[k])))



    # vanilla pytorch (with cuda if available else uses cpu)
    try:
        from pykeops.torch import Kernel, kernel_product
        
        params = {
            'id': Kernel(k + '(x,y)'),
            'gamma': 1. / (sigmac * sigmac),
            'backend': 'pytorch',
        }
        
        g0 = kernel_product(params, xc, yc, bc, mode='sum').cpu()
        speed_pytorch[k] = np.array(timeit.repeat("g0 = kernel_product(params, xc, yc, bc, mode='sum')", globals=globals(), repeat=1000, number=4)) / 4
        print('Time for Pytorch:             {:.4f}s'.format(np.median(speed_pytorch[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g0.numpy() - gnumpy)),')')
    except:
        print('Time for Pytorch:             Not Done')
    
    
    
    # keops + pytorch : generic tiled implementation (with cuda if available else uses cpu)
    try:
        from pykeops.torch import Kernel, kernel_product
    
        params = {
            'id': Kernel(k + '(x,y)'),
            'gamma': 1. / (sigmac * sigmac),
            'backend': 'auto',
        }
        g1 = kernel_product(params, xc, yc, bc,  mode='sum').cpu()
        speed_pykeops[k] = np.array(timeit.repeat("g1 = kernel_product(params, xc, yc, bc, mode='sum')", globals=globals(), repeat=100, number=4)) / 4
        print('Time for keops generic:       {:.4f}s'.format(np.median(speed_pykeops[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g1.data.numpy() - gnumpy)), ')')
    except:
        print('Time for keops generic:       Not Done')
    
    
    
    # specific cuda tiled implementation (if cuda is available)
    try:
        from pykeops.numpy import RadialKernelConv
        my_conv = RadialKernelConv(type)
        g2 = my_conv(x, y, b, sigma, kernel=k)
        speed_pykeops_specific[k] = np.array(timeit.repeat('g2 = my_conv(x, y, b, sigma, kernel=k)', globals=globals(), repeat=100, number=4)) / 4
        print('Time for keops cuda specific: {:.4f}s'.format(np.median(speed_pykeops_specific[k])), end='')
        print('   (absolute error:       ', np.max(np.abs(g2 - gnumpy)),')')
    except:
        print('Time for keops cuda specific: Not Done')


####################################################################
# display results

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

plt.legend(fake_handles, ['numpy', 'pytorch', 'pykeops', 'pykeos specific'], loc='upper center')

plt.show()
