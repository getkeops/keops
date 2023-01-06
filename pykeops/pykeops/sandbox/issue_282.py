import torch
from pykeops.torch import Vi, Vj


def GaussGradKernel_ref(gam, D):
    x, y = Vi(0, D), Vj(1, D)
    D2 = x.sqdist(y)
    K = (-D2 * gam / 2).exp()
    yo = -gam * K * (x - y)
    return yo.sum_reduction(axis=1)


def GaussGradKernel(gam, D):
    x, y = Vi(0, D), Vj(1, D)
    D2 = x.sqdist(y)
    K = (-D2 * (gam / 2)).exp()
    yo = -gam * K * (x - y)
    return yo.sum_reduction(axis=1)


gam = 1.0
D = 3
G_ref = GaussGradKernel_ref(gam, D)
G = GaussGradKernel(gam, D)

N = 10
x = torch.rand(N, D)
y = torch.rand(N, D)

G_ref(x, y)
G(x, y)
