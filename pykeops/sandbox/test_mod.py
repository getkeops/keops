import pykeops
import torch
import math
from pykeops.torch import LazyTensor

def torch_mod(input, modulus, offset=0):
    return input - modulus * torch.floor((input - offset)/modulus)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

offset = -0.1#-math.pi/2

x = torch.rand(10000, 1)*2*math.pi
y = x.data.clone()
x = x.to(device)
y = y.to(device)
x.requires_grad = True
y.requires_grad = True

x_i = LazyTensor(x[:, None])
s1 = x_i.mod(math.pi, offset).sum(0)
s2 = torch.sum(torch_mod(y, math.pi, offset))
print("s1 - s2", torch.abs(s1 - s2).item())
assert torch.abs(s1 - s2) < 1e-3, torch.abs(s1 - s2)

s1.backward()
s2.backward()

print("grad_s1 - grad_s2", torch.max(torch.abs(x.grad - y.grad)).item())
assert torch.max(torch.abs(x.grad - y.grad)) < 1e-3