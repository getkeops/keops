import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import numpy as np
from pykeops.torch.kernels import Kernel, kernel_product
try:
    from pykeops.numpy.convolutions.radial_kernels import radial_kernels_conv
except:
    pass

import time, timeit

N = 2000 ; M = 300; D = 3; E = 3

# declare numpy 
x = np.random.randn(N,D).astype('float32')
y = np.random.randn(M,D).astype('float32')
b = np.random.randn(M,E).astype('float32')
s = np.array([2.4]).astype('float32')

def np_kernel(x, y, s, kernel) :
    sq = np.sum( (x[:,np.newaxis,:] - y[np.newaxis,:,:]) **2, axis=2)
    if   kernel == "gaussian"  : return np.exp( -sq / (s*s))
    elif kernel == "laplacian" : return np.exp( -np.sqrt(sq) / s)
    elif kernel == "cauchy"    : return  1. / ( 1 + sq / (s*s) )
    elif kernel == "inverse_multiquadric" : return np.sqrt(  1. / ( s*s + sq ) )

# declare the torch counterpart
try:
    import torch

    use_cuda = torch.cuda.is_available()
    dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    xc = torch.from_numpy(x.copy()).type(dtype)
    yc = torch.from_numpy(y.copy()).type(dtype)
    bc = torch.from_numpy(b.copy()).type(dtype)
    sc = torch.from_numpy(s.copy()).type(dtype)

    def torch_kernel(x, y, s, kernel) :
        sq = torch.sum( (x[:,None]-y[None])**2 , 2 ) 
        if   kernel == "gaussian"  : return torch.exp( -sq / (s*s))
        elif kernel == "laplacian" : return torch.exp( -torch.sqrt(sq) /s)
        elif kernel == "cauchy"    : return  1. / ( 1 + sq / (s*s) )
        elif kernel == "inverse_multiquadric"    : return torch.sqrt(  1. / ( s*s + sq ) )
except:
    pass

# Define a kernel:
# Wrap it (and its parameters) into a JSON dict structure
sigma = torch.autograd.Variable(torch.Tensor([2.4]), requires_grad=False).type(dtype)
mode = "sum"

##############################
# Benchmark
##############################

enable_GC = False # Garbage collection?
GC = 'gc.enable();' if enable_GC else 'pass;'
LOOPS = 200
print("Times to compute ", LOOPS, " convolutions of size {}x{}:".format(N,M))
print("\n",end="")

for k in (["gaussian", "laplacian", "cauchy", "inverse_multiquadric"]):
    print(k, " kernel: -----------------------------------")

    # generic tiled implementation
    kernel = Kernel(k+"(x,y)")
    params = {
        "id"      : kernel,
        "gamma"   : 1./sigma**2,
        "backend" : "auto",
    }
    g1 = kernel_product( xc,yc,bc, params, mode=mode).cpu()
    speed_pykeops_gen = timeit.Timer('g1 = kernel_product( xc,yc,bc, params, mode=mode).cpu()', GC,  globals = globals(), timer = time.time).timeit(LOOPS)
    print("Time for keops generic: {:.4f}s".format(speed_pykeops_gen))
    
    # pure numpy
    g3 =  np_kernel(x,y,s,kernel=k) @ b
    speed_numpy = timeit.Timer('g3 =  np_kernel(x,y,s,kernel=k) @ b', 
            GC, globals = globals(),
            timer = time.time).timeit(LOOPS)
    print("Time for Python:       {:.4f}s".format(speed_numpy))
    print("Absolute error:       ", np.max(np.abs(g1.data.numpy() - g3)), "\n")

    # pytorch (with cuda if available else uses cpu)
    try:
        g0 = torch.mm(torch_kernel(xc,yc,sc,kernel=k),bc).cpu().numpy()
        speed_pytorch = timeit.Timer('g0 = torch.mm(torch_kernel(xc,yc,sc,kernel=k),bc)#.cpu().numpy()', GC, globals = globals(), timer = time.time).timeit(LOOPS)
        print("Time for Pytorch: {:.4f}s".format(speed_pytorch))
        print("Absolute error:       ", np.max(np.abs(g0 - g3)),"\n")
    except:
        pass

    # cuda tiled implementation (if cuda is available)
    try:
        g2 = np.zeros([N,E]).astype('float32') ; radial_kernels_conv(x, y, b, g2, s, kernel=k)
        g2 = np.zeros([N,E]).astype('float32')
        speed_pykeops = timeit.Timer('radial_kernels_conv(x, y, b, g2, s, kernel=k)', GC, globals = globals(), timer = time.time).timeit(LOOPS)
        print("Time for keops cuda specific: {:.4f}s".format(speed_pykeops))
        print("Absolute error:       ", np.max(np.abs(g2 - g3)),"\n")
    except:
        pass


