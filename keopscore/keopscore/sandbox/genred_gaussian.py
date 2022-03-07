# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import Genred

M, N, D, DV = 20, 30, 3, 1

dtype = torch.float64

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
do_warmup = False

x = torch.rand(M, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)

aliases = [f"x=Vi(0,{D})", f"y=Vj(1,{D})", f"b=Vj(2,{DV})"]
formula = "Exp(-Sum(Square(x-y)))*b"

fun = Genred(formula, aliases, reduction_op="Sum", axis=1, sum_scheme="block_sum")

if do_warmup:
    fun(x[: min(M, 100), :], y[: min(N, 100), :], b[: min(N, 100), :])
    fun(x[: min(M, 100), :], y[: min(N, 100), :], b[: min(N, 100), :])

for k in range(10):
    start = time.time()
    fun(x, y, b)
    end = time.time()
    print("time for genred:", end - start)
