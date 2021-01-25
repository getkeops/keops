from operations import *
from reductions import *
from link_compile import GpuReduc1D
import time
import os

D = 3

start = time.time()
x = Var(0,D,0)
y = Var(1,D,1)
a = Var(2,D,0)
z = x+y #Exp(-Sum(Square(x-y)))*a
f = Sum_Reduction(z,0)
g = Grad(f,x)
h = Grad(g,x)
k = Grad(h,x)
l = Grad(k,x)
nargs = 6
myred = GpuReduc1D(f,dtype="float",dtypeacc="float",nargs=nargs)

myred.compile_code()
elapsed = time.time() - start
print("time for compile : ",elapsed)



import torch

start = time.time()
N = 10
x = torch.zeros(N, D).cuda()
y = torch.ones(N, D).cuda()
a = torch.rand(N, D).cuda()
b = torch.ones(N, D).cuda()
c = torch.ones(N, D).cuda()
d = torch.ones(N, D).cuda()
out = torch.zeros(N,D).cuda()
elapsed = time.time() - start
print(f"time for init tensors : {elapsed}")
    
for k in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    myred(N, N, out, x, y, a, b, c, d)
    end.record()
    torch.cuda.synchronize()
    print(f"time for eval (run {k}) : {start.elapsed_time(end)/1000}")

import torch
for k in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    x_ = (x[:,None,:])
    y_ = (y[None,:,:])
    a_ = (a[:,None,:])
    K = (-((x_-y_)**2).sum(dim=2)).exp()
    z_ = x_+y_#K[:,:,None]*a_
    out_ = z_.sum(dim=0)
    end.record()
    torch.cuda.synchronize()
    print(f"time for torch (run {k}) : {start.elapsed_time(end)/1000}")

print(out)
print(out_)
print((torch.norm(out_-out)/torch.norm(out_)).item())







