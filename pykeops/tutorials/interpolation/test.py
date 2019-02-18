

import numpy as np
import torch
from pykeops.torch import Genred 

from linsolve import KernelLinearSolver, InvLinOp

dtype = 'float64'  # May be 'float32' or 'float64'
useGpu = "auto"   # may be True, False or "auto"
backend = torch   # np or torch
 
def GaussKernel(D,Dv,sigma):
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1)
    oos2 = torch.Tensor([1.0/sigma**2])
    def K(x,y,b):
        return my_routine(x,y,b,oos2)
    return K

def GaussKernelMatrix(sigma):
    oos2 = 1.0/sigma**2
    def f(x,y):
        D = x.shape[1]
        sqdist = 0
        for k in range(D):
            sqdist += (x[:,k][:,None]-(y[:,k][:,None]).t())**2
        return torch.exp(-oos2*sqdist)
    return f

arraysum = lambda a : torch.dot(a.view(-1), torch.ones_like(a).view(-1))
grad = torch.autograd.grad

        
D = 2
N = 4
sigma = .1
x = torch.rand(N, D, requires_grad=True)
b = torch.rand(N, D, requires_grad=True)
K = GaussKernel(D,D,sigma)
def Kinv(x,b):
    def KernelLinOp(a):
        return K(x,x,a)
    return InvLinOp(KernelLinOp,b,x)
c = Kinv(x,b)
s = arraysum(c*c)
u, = grad(s,x,create_graph=True)
print("u=",u)


###

xx = x.clone()
bb = b.clone()
MM = GaussKernelMatrix(sigma)(xx,xx)
cc = torch.gesv(bb,MM)[0].contiguous()
ss= arraysum(cc*cc)
uu, = grad(ss,xx,create_graph=True)
print("uu=",uu)   

 

###


xxx = x.clone()
bbb = b.clone()
MMM= GaussKernelMatrix(sigma)(xxx,xxx)
MMMi = torch.inverse(MMM)
ccc = MMMi@bbb
sss = arraysum(ccc*ccc)
uuu, = grad(sss,xxx,create_graph=True)
print("uuu=",uuu)  

###

xxxx = x.clone()
bbbb = b.clone()
MMMM= GaussKernelMatrix(sigma)(xxxx,xxxx)
MMMMi = torch.inverse(MMMM)
cccc = MMMMi@bbbb
eeee = MMMMi@cccc
uuuu = grad(MMMM@cccc.data,xxxx,-2*eeee)[0]
print("uuuu=",uuuu)  


print("2nd order derivative")

e = torch.randn(N,D)
v = grad(u,x,e)
print("v=",v)

ee = e.clone()
vv = grad(uu,xx,ee)
print("vv=",vv)

eee = e.clone()
vvv = grad(uuu,xxx,eee)
print("vvv=",vvv)

# s=|c|^2=|Mib|^2
# ds = 2<Mib,dMi b>
#   = -2<c,Mi dM Mi b>
# ds.dx = -2<c,Mi dM.dx Mi b>
#      = -2<c,Mi d(Mc).dx>
#      = -2<d(Mc)^T Mi c,dx>


# c = Kinv(x,b)
# M = GaussKernelMatrix(sigma)(x,x)
# e = torch.gesv(c,M)[0]
# Mi = torch.inverse(M)
# cc = Mi@b
# r = torch.rand(N,D)
# print("grad(Mi@c)=",grad(Kinv(x,c),x,r))
# print("grad(Mi@cc)=",grad(Mi@cc,x,r))
# e = Mi@c
# print("M@c.data=",M@c.data)
# print("e=",e)
# print(grad(M@c.data,x,-2*e)[0])
#
# print("2nd order derivative")
#
# ss = arraysum(u*u)
# tt = arraysum(v*v)
# uu = torch.autograd.grad(ss,x,create_graph=True)
# vv = torch.autograd.grad(tt,x,create_graph=True)
# print(uu)
# print(vv)
