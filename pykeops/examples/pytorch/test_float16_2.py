# Test for half precision support in KeOps
# We perform a gaussian convolution with half, single and double precision
# and compare timings and accuracy
import time

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
    x_i = LazyTensor( x[:,None,:] )
    y_j = LazyTensor( y[None,:,:] )  
    b_j = LazyTensor( b[None,:,:] ) 
    p = LazyTensor( p ) 
    D_ij = (x_i+y_j).min(axis=2)
    K_ij = ((- p*D_ij) * b_j)
    K_ij = K_ij.sum(axis=1,call=False,**kwargs)
    return K_ij

M, N, D = 4, 4, 4#1000000, 1000000, 3

if backend == "torch":
    torch.manual_seed(1)
    x = torch.randn(M, D, dtype=torch.float64).cuda(device_id)
    y = torch.randn(N, D, dtype=torch.float64).cuda(device_id)
    b = torch.randn(N, 1, dtype=torch.float64).cuda(device_id)
    p = torch.randn(1, dtype=torch.float64).cuda(device_id)
    xf = x.float()
    yf = y.float()
    bf = b.float()
    pf = p.float()
    xh = x.half()
    yh = y.half()
    bh = b.half()
    ph = p.half()
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

Ntest_half, Ntest_float = 1, 1

# computation using float32
K_keops32 = K(xf,yf,bf,pf)
res_float = K_keops32()
res_float = res_float[:100,:];
print("comp float, time : ",timeit.timeit("K_keops32()",number=Ntest_float,setup="from __main__ import K_keops32"))
print(res_float)

# computation using float16
K_keops16 = K(xh[:100,:],yh,bh,ph)
K_ij = K_keops16()
res_half = K_ij
print("comp half, time : ",timeit.timeit("K_keops16()",number=Ntest_half,setup="from __main__ import K_keops16"))
print(res_half)

if backend == "torch":
    print("relative mean error half vs float : ",((res_half.float()-res_float).abs().mean()/res_float.abs().mean()).item())
    print("relative max error half vs float : ",((res_half.float()-res_float).abs().max()/res_float.abs().mean()).item())

