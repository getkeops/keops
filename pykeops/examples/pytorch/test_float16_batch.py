# Test for half precision support in KeOps
# We perform a gaussian convolution with half, single and double precision
# and compare timings and accuracy
import GPUtil
from threading import Thread
import time


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

backend = "torch"  # "torch" or "numpy", but only "torch" works for now
device_id = 0
if backend == "torch":
    import torch
    from pykeops.torch import LazyTensor
else:
    import numpy as np
    from pykeops.numpy import LazyTensor

import timeit

def K(x,y,b,p,**kwargs):
    x_i = LazyTensor( x[:,:,:,None,:] )
    y_j = LazyTensor( y[:,:,None,:,:] )  
    b_j = LazyTensor( b[:,:,None,:,:] ) 
    p = LazyTensor( p[:,:,None,None,:] ) 
    D_ij = ((x_i - y_j)**2).sum(axis=4)  
    K_ij = (- p*D_ij).exp() * b_j             
    K_ij = K_ij.sum(axis=3,call=False,**kwargs)
    return K_ij

B, C, M, N, D = 3, 5, 11, 11, 1

if backend == "torch":
    torch.manual_seed(1)
    x = torch.randn(B, C, M, D, dtype=torch.float64).cuda(device_id)
    y = torch.randn(B, C, N, D, dtype=torch.float64).cuda(device_id)
    b = torch.randn(B, C, N, 1, dtype=torch.float64).cuda(device_id)
    p = torch.randn(B, C, D, dtype=torch.float64).cuda(device_id)
    xf = x.float()
    yf = y.float()
    bf = b.float()
    pf = p.float()
    xh = x.half()
    yh = y.half()
    bh = b.half()
    ph = p.half()
else:
    x = np.random.randn(B, C, M, D)
    y = np.random.randn(B, C, N, D)
    b = np.random.randn(B, C, N, 1)
    xf = x.astype(np.float32)
    yf = y.astype(np.float32)
    bf = b.astype(np.float32)
    xh = x.astype(np.float16)
    yh = y.astype(np.float16)
    bh = b.astype(np.float16)

Ntest_half, Ntest_float = 0, 0
# monitor = Monitor(1e-6)
# computation using float32
K_keops32 = K(xf,yf,bf,pf)
res_float = K_keops32()
print("comp float, time : ",timeit.timeit("K_keops32()",number=Ntest_float,setup="from __main__ import K_keops32"))
#print(res_float)
# monitor.stop()

# computation using float16
# monitor = Monitor(1e-6)
K_keops16 = K(xh,yh,bh,ph)
K_ij = K_keops16()
res_half = K_ij
print("comp half, time : ",timeit.timeit("K_keops16()",number=Ntest_half,setup="from __main__ import K_keops16"))
#print(res_half)
# monitor.stop()

if backend == "torch":
    print("relative mean error half / float : ",((res_half.float()-res_float).abs().mean()/res_float.abs().mean()).item())
    print("relative max error half / float : ",((res_half.float()-res_float).abs().max()/res_float.abs().mean()).item())
else:
    print("relative mean error half / float : ",(np.mean(np.abs(res_half.astype(np.float64)-res_float))/np.mean(np.abs(res_float))).item())
    print("relative max error half / float : ",(np.max(np.abs(res_half.astype(np.float64)-res_float))/np.mean(np.abs(res_float))).item())


