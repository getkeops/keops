
import torch
import numpy as np
grad = torch.autograd.grad

def ConjugateGradientSolver(linop,b,eps=1e-6):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    arraytype = type(b)
    copy = np.copy if arraytype == np.ndarray else torch.clone
    a = 0
    r = copy(b)
    nr2 = (r**2).sum()
    if nr2 < eps**2:
        return 0*r
    p = copy(r)
    k = 0
    while True:
        Mp = linop(p)
        alpha = nr2/(p*Mp).sum()
        a += alpha*p
        r -= alpha*Mp
        nr2new = (r**2).sum()
        if nr2new < eps**2:
            break
        p = r + (nr2new/nr2)*p
        nr2 = nr2new
        k += 1
    #print("numiters=",k)
    return a

def f(x,a):
    return a*x**3

class g_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b):
        a = b/x**3
        ctx.save_for_backward(a,x)
        return a
    @staticmethod
    def backward(ctx, grad_output):
        a,x = ctx.saved_tensors
        abar = torch.tensor(a.data,requires_grad=True)
        with torch.enable_grad():
            e = g(x,grad_output)
            gfx = grad(f(x,abar),x,create_graph=True)[0]
            return -e*link(gfx,x,abar,a), e

class link_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx,f,x,abar,a):
        ctx.save_for_backward(f,x,abar,a)
        return f
    @staticmethod
    def backward(ctx, grad_output):
        f,x,abar,a = ctx.saved_tensors
        gx,ga = grad(f,[x,abar],grad_output,create_graph=True,allow_unused=True)
        return None, link(gx,x,abar,a), link(ga,x,abar,a), link(ga,x,abar,a)

g = g_Impl.apply
def link(*args):
    if args[0] is None:
        return None
    else:
        return link_Impl.apply(*args)

x = torch.randn(1,requires_grad=True)
b = torch.randn(1)

gx = grad(g(x,b),x,create_graph=True)[0]
gx_ = grad(b/x**3,x,create_graph=True)[0]
print("gx = ",gx.data)
print("gx_ = ",gx_.data)

ggx = grad(gx,x,create_graph=True)[0]
ggx_ = grad(gx_,x,create_graph=True)[0]
print("ggx = ",ggx.data)
print("ggx_ = ",ggx_.data)

gggx = grad(ggx,x,create_graph=True)[0]
gggx_ = grad(ggx_,x,create_graph=True)[0]
print("gggx = ",gggx.data)
print("gggx_ = ",gggx_.data)

ggggx = grad(gggx,x,create_graph=True)[0]
ggggx_ = grad(gggx_,x,create_graph=True)[0]
print("ggggx = ",ggggx.data)
print("ggggx_ = ",ggggx_.data)

# x = torch.randn(1,requires_grad=True)
# a = x**2
# abar = torch.tensor(a.data,requires_grad=True)
# s = x**2*a*abar # s=x^6
# gs = grad(link(s,x,abar,a),x,create_graph=True)[0] # 6x^5
# print("gs = ",gs.data)
# print("6x^5 = ",6*x.data**5)
# ggs = grad(gs,x,create_graph=True)[0] # 30*x^4
# print("ggs = ",ggs.data)
# print("30x^4 = ",30*x.data**4)
# gggs = grad(ggs,x,create_graph=True)[0] # 120*x^3
# print("gggs = ",gggs.data)
# print("120x^3 = ",120*x.data**3)

