# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import LazyTensor

B1, B2, M, N, D, DV = 3, 4, 10, 200, 3, 1000


test_grad = False
device_id = 'cuda'
do_warmup = True

x = torch.ones(B1, B2, M, 1, D, device=device_id)/math.sqrt(D)
y = torch.ones(B1, B2, 1, N, D, device=device_id)/math.sqrt(D)
b = torch.ones(B1, B2, N, DV, requires_grad=test_grad, device=device_id)

def fun(x,y,b,backend):
    if backend=="keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    elif backend!="torch":
        raise ValueError("wrong backend")
    Dxy = ((x-y)**2).sum(dim=4) 
    Kxy = (- Dxy).exp() 
    if backend=="keops":
        out = LazyTensor.__matmul__(Kxy,b,optional_flags=['-DENABLE_FINAL_CHUNKS=1','-DDIMFINALCHUNK=64'])
    else:
        out = Kxy @ b
    if device_id != 'cpu':
        torch.cuda.synchronize() 
    #print("out:",out.flatten()[:10])
    return out
    
backends = ["torch","keops"]
    
out = []
for backend in backends:
    if do_warmup:
        fun(x[:,:,:min(M,100),:,:].contiguous(),y[:,:,:,:min(N,100),:].contiguous(),b[:,:,:min(N,100),:].contiguous(),backend)
        fun(x[:,:,:min(M,100),:,:].contiguous(),y[:,:,:,:min(N,100),:].contiguous(),b[:,:,:min(N,100),:].contiguous(),backend)
    start = time.time()
    out.append(fun(x,y,b,backend).squeeze())
    end = time.time()
    print("time for "+backend+":", end-start )

if len(out)>1:
    print("relative error:", (torch.norm(out[0]-out[1])/torch.norm(out[0])).item() )

print("torch:",out[0].flatten()[:10])
print("keops:",out[1].flatten()[:10])

if test_grad:
    out_g = []
    for k, backend in enumerate(backends):
        start = time.time()
        out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [b])[0])
        end = time.time()
        print("time for "+backend+" (grad):", end-start )
    
    if len(out_g)>1:
        print("relative error grad:", (torch.norm(out_g[0]-out_g[1])/torch.norm(out_g[0])).item() )
    
