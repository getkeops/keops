import torch
from pykeops.torch import Genred

X1 = torch.randn(20, 15).cuda().requires_grad_()
X2 = torch.randn(5, 15).cuda().requires_grad_()
v = torch.randn(5, 120).cuda().requires_grad_()
sigma = torch.randn(15).cuda().requires_grad_()

formula = "Exp(SqDist(x1 / s, x2 / s) * IntInv(-2)) * v"
aliases = [
    "x1 = Vi(%d)" % (X1.shape[1]),
    "x2 = Vj(%d)" % (X2.shape[1]),
    "v = Vj(%d)" % (v.shape[1]),
    "s = Pm(%d)" % (sigma.shape[0]),
]
other_vars = [sigma]

fn = Genred(formula, aliases, reduction_op="Sum", axis=1)
out = fn(X1, X2, v, *other_vars, backend="GPU_1D")
print(out.shape)

grad = torch.autograd.grad(out.sum(), [X1])
print(grad[0].shape)
