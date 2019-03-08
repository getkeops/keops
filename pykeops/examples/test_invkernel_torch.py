import torch
import time 

from pykeops.torch.operations import InvKernelOp

D = 2
Dv = 2
N = 100
sigma = .1

# define the kernel : here a gaussian kernel
formula = 'Exp(-oos2*SqDist(x,y))*b'
variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
             'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
             'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
             'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
             
# define the inverse kernel operation : here the 'b' argument specifies that linearity is with respect to variable b in formula.
lmbda = 0.01
Kinv = InvKernelOp(formula, variables, 'b', lmbda=lmbda, axis=1)

# data
x = torch.rand(N, D, requires_grad=True)
b = torch.rand(N, D)
oos2 = torch.Tensor([1.0/sigma**2])

# apply
print("Kernel inversion operation with gaussian kernel, ",N," points in dimension ",D)
start = time.time()
c = Kinv(x,x,b,oos2)
end = time.time()
print('Time to perform (KeOps):', round(end - start, 5), 's')

# compare with direct PyTorch implementation
start = time.time()
c_ = torch.gesv(b,lmbda*torch.eye(N)+torch.exp(-torch.sum((x[:,None,:]-x[None,:,:])**2,dim=2)/sigma**2))[0]
end = time.time()
print('Time to perform (PyTorch):', round(end - start, 5), 's')
print("relative error = ",(torch.norm(c-c_)/torch.norm(c_)).item())

print("1st order derivative")
e = torch.randn(N,D)
start = time.time()
u, = torch.autograd.grad(c,x,e)
end = time.time()
print('Time to perform (KeOps):', round(end - start, 5), 's')
start = time.time()
u_, = torch.autograd.grad(c_,x,e)
end = time.time()
print('Time to perform (PyTorch):', round(end - start, 5), 's')
print("relative error = ",(torch.norm(u-u_)/torch.norm(u_)).item())



