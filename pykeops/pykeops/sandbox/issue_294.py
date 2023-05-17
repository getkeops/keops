import torch
from pykeops.torch import Genred

B = 3
X1 = torch.rand(B, 20, 1, 100).cuda().requires_grad_()
X2 = torch.rand(B, 1, 5, 100).cuda().requires_grad_()
v = torch.randn(B, 1, 5, 120).cuda().requires_grad_()
sigma = (0.5 + torch.rand(B, 1, 1, 100)).cuda().requires_grad_()

formula = "Exp(SqDist((x1 / s), (x2 / s)) * IntInv(-2)) * v"
aliases = [
    "x1 = Vi(%d)" % (X1.shape[-1]),
    "x2 = Vj(%d)" % (X2.shape[-1]),
    "v = Vj(%d)" % (v.shape[-1]),
    "s = Pm(%d)" % (sigma.shape[-1]),
]
other_vars = [sigma]

fn = Genred(formula, aliases, reduction_op="Sum", axis=1)
out = fn(X1, X2, v, *other_vars, backend="GPU_1D")
print(out.shape)

out_torch = (
    (-0.5 * ((X1 / sigma - X2 / sigma) ** 2).sum(axis=3, keepdim=True)).exp() * v
).sum(axis=2, keepdim=True)
print(out_torch.shape)

print(torch.norm(out).item())
print(torch.norm(out_torch).item())

print((torch.norm(out - out_torch) / torch.norm(out_torch)).item())
