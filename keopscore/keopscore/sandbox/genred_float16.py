# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import Genred

(
    M,
    N,
) = (
    5,
    5,
)

dtype = torch.float16

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

x = torch.zeros(M, 1, device=device_id, dtype=dtype)
b = torch.ones(N, 1, device=device_id, dtype=dtype)
y = torch.zeros(N, 1, device=device_id, dtype=dtype)
y[0] = 1
z = -5 * torch.ones(N, 1, device=device_id, dtype=dtype)

aliases = ["x=Vi(0,1)", "b=Vj(1,1)", "y=Vj(2,1)", "z=Vj(3,1)"]
formula = "SumT(y,1)"

fun = Genred(formula, aliases, reduction_op="Sum", axis=1, sum_scheme="block_sum")


for k in range(1):
    start = time.time()
    out = fun(x, b, y, z)
    end = time.time()
    print("time for genred:", end - start)
