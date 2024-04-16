from pykeops.torch import Vi, Vj, LazyTensor
import torch


xc = torch.randn(256, 5)
xc.requires_grad_(True)

x_i = Vi(xc)
x_j = Vj(xc)
d = -LazyTensor.sqdist(x_i, x_j)
fun = d.logsumexp_reduction(dim=1, call=False)

fun().sum().backward()

print("1 ok")


xc = torch.randn(256, 5)
yc = torch.randn(256, 5)
xc.requires_grad_(True)

x_i = Vi(0, 5)
x_j = Vj(1, 5)
d = -LazyTensor.sqdist(x_i, x_j)
fun = d.logsumexp_reduction(dim=1, call=False)

fun(xc, yc).sum().backward()


print("2 ok")


xc = torch.randn(256, 5)
xc.requires_grad_(True)

x_i = Vi(0, 5)
x_j = Vj(1, 5)
d = -LazyTensor.sqdist(x_i, x_j)
fun = d.logsumexp_reduction(dim=1, call=False)

fun(xc, xc).sum().backward()


print("3 ok")
