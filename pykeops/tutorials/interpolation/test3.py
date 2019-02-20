
import numpy as np
import torch
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

class g_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, linop, b, x):
        a = ConjugateGradientSolver(linop,b)
        ctx.save_for_backward(a,x)
        return a
    @staticmethod
    def backward(ctx, grad_output):
        a,x = ctx.saved_tensors
        abar = torch.tensor(a.data,requires_grad=True)
        with torch.enable_grad():
            gb = g(linop,grad_output,x)
            gfx = grad(linop(abar),x,-gb,create_graph=True)[0]
            return None, gb, link(gfx,x,abar,a)

class link_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx,f,x,abar,a):
        ctx.save_for_backward(f,x,abar,a)
        return f
    @staticmethod
    def backward(ctx, grad_output):
        f,x,abar,a = ctx.saved_tensors
        with torch.enable_grad():
            gx, ga = grad(f,[x,abar],grad_output,create_graph=True,allow_unused=True)
        return None, link(gx,x,abar,a), None, link(ga,x,abar,a)

g = g_Impl.apply
def link(*args):
    if args[0] is None:
        return None
    else:
        return link_Impl.apply(*args)

def f(x):
    def linop(a):
        return a*x**3
    return linop


x = torch.ones(1,requires_grad=True)
linop = f(x)
b = torch.ones(1)

val = g(linop,b,x)
print("val = ",val.data)
print("b/x**3 = ",b/x.data**3)

gx = grad(val,x,create_graph=True)[0]
print("gx = ",gx.data)
print("-3b/x**4 = ",-3*b/x.data**4)

ggx = grad(gx,x,create_graph=True)[0]
print("ggx = ",ggx.data)
print("12b/x**5 = ",12*b/x.data**5)

gggx = grad(ggx,x,create_graph=True)[0]
print("gggx = ",gggx.data)
print("-60b/x**6 = ",-60*b/x.data**6)

ggggx = grad(gggx,x,create_graph=True)[0]
print("ggggx = ",ggggx.data)
print("360b/x**7 = ",360*b/x.data**7)

