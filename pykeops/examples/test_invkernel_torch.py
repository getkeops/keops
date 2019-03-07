import torch
from pykeops.torch import Genred 

from pykeops.torch.invkernel import InvKernelOp

D = 2
Dv = 2
N = 4
sigma = .1

# define the kernel : here a gaussian kernel
formula = 'Exp(-oos2*SqDist(x,y))*b'
variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
             'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
             'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
             'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
             
# define the inverse kernel operation : here the 'b' argument specifies that linearity is with respect to variable b in formula.
Kinv = InvKernelOp(formula, variables, 'b')

# data
x = torch.rand(N, D, requires_grad=True)
b = torch.rand(N, D)
oos2 = torch.Tensor([1.0/sigma**2])

# apply
print("kernel inversion operation")
c = Kinv(x,x,b,oos2)
print("c = ",c)

print("1st order derivative")
e = torch.randn(N,D)
u, = torch.autograd.grad(c,x,e,create_graph=True)
print("u=",u)
