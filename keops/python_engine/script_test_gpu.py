from operations import *
from reductions import *
from link_compile import GpuReduc1D
import time
import os

D = 2

start = time.time()
x = Var(0,2,0)
y = Var(1,2,1)
z = Exp(-Sum(Square(x-y)))
f = Sum_Reduction(z,0)
g = Grad(f,x)
h = Grad(g,x)
k = Grad(h,x)
l = Grad(k,x)
nargs = 6
myred = GpuReduc1D(f,dtype="float",dtypeacc="float",nargs=nargs)

myred.write_code()

"""
myred.compile_code()
elapsed = time.time() - start
print("time for compile : ",elapsed)



print(g)


import torch

N = 1000
x = torch.ones(N, D)
y = torch.zeros(N, D)
a = torch.ones(N, D)
b = torch.ones(N, D)
c = torch.ones(N, D)
d = torch.ones(N, D)
out = torch.zeros(N)

for k in range(10):
    start = time.time()
    myred(N, N, out, x, y, a, b, c, d)
    elapsed = time.time() - start
    print(f"time for eval (run {k}) : {elapsed}")

import torch
for k in range(10):
    start = time.time()
    x_ = (x[:,None,:])
    y_ = (y[None,:,:])
    z_ = (-((x_-y_)**2).sum(dim=2)).exp()
    out_ = z_.sum(dim=0)
    elapsed = time.time() - start
    print(f"time for torch (run {k}) : {elapsed}")


print((torch.norm(out_-out)/torch.norm(out_)).item())

"""






