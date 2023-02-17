import torch
from pykeops.torch import Genred

X1 = torch.rand(20, 100).cuda().requires_grad_()
X2 = torch.rand(5, 100).cuda().requires_grad_()
v = torch.randn(5, 120).cuda().requires_grad_()
sigma = (.5+torch.rand(100)).cuda().requires_grad_()

formula = 'Exp(SqDist((x1 / s), (x2 / s)) * IntInv(-2)) * v'
aliases = [
    'x1 = Vi(%d)' % (X1.shape[1]),
    'x2 = Vj(%d)' % (X2.shape[1]),
    'v = Vj(%d)' % (v.shape[1]),
    's = Pm(%d)' % (sigma.shape[0])
]
other_vars = [sigma]

fn = Genred(formula, aliases, reduction_op="Sum", axis=1)
out = fn(X1, X2, v, *other_vars, backend="GPU_1D")
print(out.shape)

out_torch = (-.5*((X1[:,None,:]/sigma[None,None,:]-X2[None,:,:]/sigma[None,None,:])**2).sum(axis=2)).exp() @ v
print(out_torch.shape)

print(torch.norm(out).item())
print(torch.norm(out_torch).item())

print((torch.norm(out-out_torch)/torch.norm(out_torch)).item())