

import numpy as np
import torch
from pykeops.torch import Genred 

from pykeops.tutorials.interpolation.torch.linsolve import InvKernelOp

def InvGaussKernel(D,Dv,sigma):
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = InvKernelOp(formula, variables, 'b', axis=1)
    oos2 = torch.Tensor([1.0/sigma**2])
    def Kinv(x,b):
        return my_routine(x,x,b,oos2)
    return Kinv

arraysum = lambda a : torch.dot(a.view(-1), torch.ones_like(a).view(-1))
grad = torch.autograd.grad
     
D = 2
N = 4
sigma = .1
x = torch.rand(N, D, requires_grad=True)
b = torch.rand(N, D)
Kinv = InvGaussKernel(D,D,sigma)
c = Kinv(x,b)
print("c = ",c)

print("1st order derivative")

e = torch.randn(N,D)
u, = grad(c,x,e,create_graph=True)
print("u=",u)

print("2nd order derivative")

e = torch.randn(N,D)
v = grad(u,x,e,create_graph=True)[0]
print("v=",v)

print("3rd order derivative")

e = torch.randn(N,D)
w = grad(v,x,e,create_graph=True)[0]
print("w=",w)


