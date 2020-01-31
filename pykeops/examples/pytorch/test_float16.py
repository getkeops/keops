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

def K(x,y,b,**kwargs):
    x_i = LazyTensor( x[:,None,:] )
    y_j = LazyTensor( y[None,:,:] )  
    b_j = LazyTensor( b[None,:,:] ) 
    D_ij = ((x_i - y_j)**2).sum(axis=2)  
    K_ij = (- D_ij).exp() * b_j             
    K_ij = K_ij.sum(axis=1,call=False,**kwargs)
    print(K_ij)
    return K_ij

M, N, D = 1000, 100000, 3

if backend == "torch":
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=torch.float64).cuda(device_id)
    y = torch.randn(N, D, dtype=torch.float64).cuda(device_id)
    b = torch.randn(N, 1, dtype=torch.float64).cuda(device_id)
    xf = x.float()
    yf = y.float()
    bf = b.float()
    xh = x.half()
    yh = y.half()
    bh = b.half()
else:
    x = np.random.randn(M, D)
    y = np.random.randn(N, D)
    b = np.random.randn(N, 1)
    xf = x.astype(np.float32)
    yf = y.astype(np.float32)
    bf = b.astype(np.float32)
    xh = x.astype(np.float16)
    yh = y.astype(np.float16)
    bh = b.astype(np.float16)

Ntest_half, Ntest_float, Ntest_double = 10, 10, 10
# monitor = Monitor(1e-6)
# computation using float32
K_keops32 = K(xf,yf,bf)
res_float = K_keops32()
print("comp float, time : ",timeit.timeit("K_keops32()",number=Ntest_float,setup="from __main__ import K_keops32"))
# monitor.stop()

# computation using float64
# monitor = Monitor(1e-6)
K_keops64 = K(x,y,b)
res_double = K_keops64()
print("comp double, time : ",timeit.timeit("K_keops64()",number=Ntest_double,setup="from __main__ import K_keops64"))
# monitor.stop()
if backend == "torch":
    print("relative mean error float / double : ",(res_float.double()-res_double).abs().mean()/res_double.abs().mean())
    print("relative max error float / double : ",(res_float.double()-res_double).abs().max()/res_double.abs().mean())
else:
    print("relative mean error float / double : ",np.mean(np.abs(res_float.astype(np.float64)-res_double))/np.mean(np.abs(res_double)))
    print("relative max error float / double : ",np.max(np.abs(res_float.astype(np.float64)-res_double))/np.mean(np.abs(res_double)))

# computation using float16
# monitor = Monitor(1e-6)
K_keops16 = K(xh,yh,bh)
res_half = K_keops16()
print("comp half, time : ",timeit.timeit("K_keops16()",number=Ntest_half,setup="from __main__ import K_keops16"))
# monitor.stop()

if backend == "torch":
    print("relative mean error half / double : ",(res_half.double()-res_double).abs().mean()/res_double.abs().mean())
    print("relative max error half / double : ",(res_half.double()-res_double).abs().max()/res_double.abs().mean())
else:
    print("relative mean error half / double : ",np.mean(np.abs(res_half.astype(np.float64)-res_double))/np.mean(np.abs(res_double)))
    print("relative max error half / double : ",np.max(np.abs(res_half.astype(np.float64)-res_double))/np.mean(np.abs(res_double)))


