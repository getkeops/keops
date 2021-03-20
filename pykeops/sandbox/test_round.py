import pykeops
import torch
import math
from pykeops.torch import LazyTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

# rounds to the nearest integer (0 decimal)
x = torch.FloatTensor(1000, 1).uniform_(-10, 10)
y = x.data.clone()
x = x.to(device)
y = y.to(device)
x.requires_grad = True
y.requires_grad = True

x_i = LazyTensor(x[:, None])
s1 = x_i.round().sum(0)
s2 = torch.sum(torch.round(y))
print("s1 - s2", torch.abs(s1 - s2).item())
assert torch.abs(s1 - s2) < 1e-3, torch.abs(s1 - s2)

s1.backward()
s2.backward()

print("grad_s1 - grad_s2", torch.max(torch.abs(x.grad - y.grad)).item())
assert torch.max(torch.abs(x.grad - y.grad)) < 1e-3

# rounds to 3 decimal places
x = torch.FloatTensor(1000, 1).uniform_(-1, 1)
y = x.data.clone()
x = x.to(device)
y = y.to(device)
x.requires_grad = True
y.requires_grad = True

x_i = LazyTensor(x[:, None])
s1 = x_i.round(3).sum(0)
s2 = torch.sum(torch.round(y * 1e3) * 1e-3)
print("s1 - s2", torch.abs(s1 - s2).item())
assert torch.abs(s1 - s2) < 1e-3, torch.abs(s1 - s2)

s1.backward()
s2.backward()

print("grad_s1 - grad_s2", torch.max(torch.abs(x.grad - y.grad)).item())
assert torch.max(torch.abs(x.grad - y.grad)) < 1e-3
