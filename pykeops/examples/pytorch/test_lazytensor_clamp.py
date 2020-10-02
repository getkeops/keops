# Test for Clamp operation using LazyTensors

import torch
from pykeops.torch import LazyTensor

x = torch.randn(1000,   1, 3, requires_grad=True)
y = torch.randn(   1, 500, 1, requires_grad=True)
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
    Kxy = (- Dxy).exp() 
    return Kxy.sum(dim=1)
    
out_torch = fun(x,y,a,b,"torch")
out_keops = fun(x,y,a,b,"keops").squeeze()
print("relative error:", (torch.norm(out_torch-out_keops)/torch.norm(out_torch)).item() )

g_torch = torch.autograd.grad((out_torch ** 2).sum(), [x])[0] 
g_keops = torch.autograd.grad((out_keops ** 2).sum(), [x])[0] 
print("relative error grad:", (torch.norm(g_torch-g_keops)/torch.norm(g_torch)).item() )
