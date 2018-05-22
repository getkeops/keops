import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import numpy as np
import time, timeit

# is there a working GPU around ?
import GPUtil
try:
    gpu_available = len(GPUtil.getGPUs()) > 0
except:
    gpu_available = False

N = 1500 ; M = 300; D = 3; E = 3

# declare numpy 
from pykeops.numpy.utils import differences, squared_distances,grad_np_kernel, chain_rules
a = np.random.rand(N,E).astype('float32')
x = np.random.rand(N,D).astype('float32')
y = np.random.rand(M,D).astype('float32')
b = np.random.rand(M,E).astype('float32')
sigma = np.array([0.4]).astype('float32')

# declare the torch counterpart
try:
    import torch
    from torch.autograd import Variable, grad
    from pykeops.torch import Kernel, kernel_product

    use_cuda = torch.cuda.is_available()
    dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    ac = Variable(torch.from_numpy(a.copy()).type(dtype), requires_grad=True).type(dtype)
    xc = Variable(torch.from_numpy(x.copy()).type(dtype), requires_grad=True).type(dtype)
    yc = Variable(torch.from_numpy(y.copy()).type(dtype), requires_grad=True).type(dtype)
    bc = Variable(torch.from_numpy(b.copy()).type(dtype), requires_grad=True).type(dtype)
    sigmac = torch.autograd.Variable(torch.from_numpy(sigma.copy()).type(dtype), requires_grad=False).type(dtype)

except:
    pass

##############################
# Benchmark
##############################

enable_GC = False # Garbage collection?
GC = 'gc.enable();' if enable_GC else 'pass;'
LOOPS = 100
print("Times to compute ", LOOPS, " grad-convolutions of size {}x{}:".format(N,M))
print("\n",end="")

for k in (["gaussian", "laplacian", "cauchy", "inverse_multiquadric"]):
    print(k, " kernel: -----------------------------------")

    # pure numpy
    gnumpy = chain_rules(a,x,y,grad_np_kernel(x,y,sigma,kernel=k),b)
    speed_numpy = timeit.Timer('gnumpy = chain_rules(a,x,y,grad_np_kernel(x,y,sigma,kernel=k),b)',
            GC, globals = globals(),
            timer = time.time).timeit(LOOPS)
    print("Time for numpy:            {:.4f}s".format(speed_numpy))

    # cuda tiled implementation (if cuda is available)
    if gpu_available:
        try:
            from pykeops.numpy.convolutions.radial_kernels_grad1 import radial_kernels_grad1conv

            g1 = np.zeros([N,E]).astype('float32') ; radial_kernels_grad1conv(a, x, y, b, g1, sigma, kernel=k)
            g11 = np.zeros([N,E]).astype('float32')
            speed_pykeops = timeit.Timer('radial_kernels_grad1conv(a, x, y, b, g11, sigma, kernel=k)', GC, globals = globals(), timer = time.time).timeit(LOOPS)
            print("Time for keops specific:   {:.4f}s".format(speed_pykeops),end="")
            print("   (absolute error:       ", np.max(np.abs (g1 - gnumpy)), ")")
        except:
            pass
    else:
        print("No Gpu detetcted ; skipping 'specific' KeOps computation.")

    # keops + pytorch : generic tiled implementation (with cuda if available else uses cpu)
    try:
        # Define a kernel: Wrap it (and its parameters) into a JSON dict structure
        mode = "sum"
        kernel = Kernel(k+"(x,y)")
        params = {
            "id"      : kernel,
            "gamma"   : 1./sigmac**2,
            "backend" : "auto",
        }

        aKxy_b = torch.dot(ac.view(-1), kernel_product( params, xc,yc,bc, mode=mode).view(-1))
        g3   = torch.autograd.grad(aKxy_b, xc, create_graph=False)[0].cpu()
        speed_keops = timeit.Timer('g3 = torch.autograd.grad(torch.dot(ac.view(-1), kernel_product( params, xc,yc,bc, mode=mode).view(-1)), xc, create_graph=False)[0]', GC, globals = globals(), timer = time.time).timeit(LOOPS)
        print("Time for Keops+pytorch:    {:.4f}s".format(speed_keops),end="")
        print("   (absolute error:       ", np.max(np.abs(g3.data.numpy() - gnumpy)), ")")
    except:
        pass

    # vanilla pytorch (with cuda if available else uses cpu)
    try:
        # Define a kernel: Wrap it (and its parameters) into a JSON dict structure
        mode = "sum"
        kernel = Kernel(k+"(x,y)")
        params = {
            "id"      : kernel,
            "gamma"   : 1./sigmac**2,
            "backend" : "pytorch",
        }

        aKxy_b = torch.dot(ac.view(-1), kernel_product( params, xc,yc,bc, mode=mode).view(-1))
        g3   = torch.autograd.grad(aKxy_b, xc, create_graph=False)[0].cpu()
        speed_keops = timeit.Timer('g3 = torch.autograd.grad(torch.dot(ac.view(-1), kernel_product( params, xc,yc,bc, mode=mode).view(-1)), xc, create_graph=False)[0]', GC, globals = globals(), timer = time.time).timeit(LOOPS)
        print("Time for vanilla pytorch:  {:.4f}s".format(speed_keops),end="")
        print("   (absolute error:       ", np.max(np.abs(g3.data.numpy() - gnumpy)), ")")
    except:
        pass
