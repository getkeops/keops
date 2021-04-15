import pykeops
import torch
import math
from pykeops.torch import LazyTensor

# test for modulus operation


def torch_mod(input, modulus, offset=0):
    return input - modulus * torch.floor((input - offset) / modulus)


device = "cuda" if torch.cuda.is_available() else "cpu"

offset = -math.pi / 2

x = torch.rand(10000, 1) * 2 * math.pi
y = x.data.clone()
x = x.to(device)
y = y.to(device)
x.requires_grad = True
y.requires_grad = True

x_i = LazyTensor(x[:, None])
s1 = x_i.mod(math.pi, offset).sum(0)
s2 = torch.sum(torch_mod(y, math.pi, offset))

print("relative error : ", (torch.abs(s1 - s2) / torch.abs(s2)).item())
assert torch.abs(s1 - s2) / torch.abs(s2) < 1e-3

s1.backward()
s2.backward()

print(
    "relative error grad : ", (torch.norm(x.grad - y.grad) / torch.norm(y.grad)).item()
)
assert torch.norm(x.grad - y.grad) / torch.norm(y.grad) < 1e-3
