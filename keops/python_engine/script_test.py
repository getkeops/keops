from operations import *
from reductions import *
from link_compile import CpuReduc
import time
import os

start = time.time()
x = Var(0,2,0)
y = Var(1,2,1)
f = Sum_Reduction(Exp(-Sum(Square(x-y))),0)
g = Grad(f,x)
h = Grad(g,x)
k = Grad(h,x)
l = Grad(k,x)
code = CpuReduc(l,dtype="float",dtypeacc="double")()
f = open("test.cpp","w")
f.write(code)
f.close()
elapsed = time.time()-start
print("code generation time : ", elapsed)

start = time.time()
os.system("g++ -c -O3 -DUSE_OPENMP=1 test.cpp")
elapsed = time.time()-start
print("compilation time : ", elapsed)
