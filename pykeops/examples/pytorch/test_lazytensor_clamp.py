# Test for Clamp operation using LazyTensors

import time

import torch
from pykeops.torch import LazyTensor

M, N, D = 1000, 1000, 300

x = torch.randn(M, 1, D, requires_grad=True)
y = torch.randn(1, N, D, requires_grad=True)
a = -1.5
b = 1.5

def fun(x,y,a,b,backend):
    if backend=="keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
        #a = LazyTensor(a)
        #b = LazyTensor(b)
    elif backend!="torch":
        raise ValueError("wrong backend")
    Dxy = ((x*y).clamp(a,b)).sum(dim=2) 
    Kxy = (- Dxy**2).exp() 
    return Kxy.sum(dim=1)
    
start = time.time()
out_torch = fun(x,y,a,b,"torch")
end = time.time()
print("time for torch:", end-start )

start = time.time()
out_keops = fun(x,y,a,b,"keops").squeeze()
end = time.time()
print("time for keops:", end-start )

print("relative error:", (torch.norm(out_torch-out_keops)/torch.norm(out_torch)).item() )

#g_torch = torch.autograd.grad((out_torch ** 2).sum(), [x])[0] 
#g_keops = torch.autograd.grad((out_keops ** 2).sum(), [x])[0] 
#print("relative error grad:", (torch.norm(g_torch-g_keops)/torch.norm(g_torch)).item() )
