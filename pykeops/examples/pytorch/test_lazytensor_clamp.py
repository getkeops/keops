# Test for Clamp operation using LazyTensors

import torch
from pykeops.torch import LazyTensor

x = torch.randn(1000, 1, 1, requires_grad=True).cuda()
y = -1.5
z = 1.5

def fun(x,y,z,backend):
    if backend=="keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
        z = LazyTensor(z)
    elif backend!="torch":
        raise ValueError("wrong backend")
    Dxy = (x.clamp(y,z)).sum(dim=2) 
    Kxy = (- Dxy).exp() 
    return Kxy.sum(dim=1)
    
out_torch = fun(x,y,z,"torch")
out_keops = fun(x,y,z,"keops").squeeze()
print("relative error:", (torch.norm(out_torch-out_keops)/torch.norm(out_torch)).item() )

g_torch = torch.autograd.grad((out_torch ** 2).sum(), [x])[0] 
g_keops = torch.autograd.grad((out_keops ** 2).sum(), [x])[0] 
print("relative error grad:", (torch.norm(g_torch-g_keops)/torch.norm(g_torch)).item() )
