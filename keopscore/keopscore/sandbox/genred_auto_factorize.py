import time

import math
import torch
from pykeops.torch import Genred

M, N, D, DV = 200000, 300000, 1, 1

dtype = torch.float32

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

x = torch.rand(M, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)
e = torch.randn(N, DV, device=device_id, dtype=dtype)

aliases = [f"x=Vi(0,{D})", f"y=Vj(1,{D})", f"b=Vj(2,{DV})", f"e=Vi(3,{DV})"]

formula1 = formula2 = "Exp(-SqDist(x,y))*b"

order = 5
for k in range(order):
    formula1 = f"Grad({formula1},x,e)"
    formula2 = f"AutoFactorize(Grad({formula2},x,e))"

fun1 = Genred(formula1, aliases, reduction_op="Sum", axis=1, sum_scheme="block_sum")
fun2 = Genred(formula2, aliases, reduction_op="Sum", axis=1, sum_scheme="block_sum")

S = []
E = []
ntry = 10
for k in range(ntry):
    start = time.time()
    res1 = fun1(x, y, b, e)
    end = time.time()
    time1 = end - start
    start = time.time()
    res2 = fun2(x, y, b, e)
    end = time.time()
    time2 = end - start
    S.append(time1 / time2)
    E.append(torch.norm(res1 - res2) / torch.norm(res1))
print("mean error of autofact:", sum(E) / ntry)
print("mean speedup factor of autofact:", sum(S) / ntry)
