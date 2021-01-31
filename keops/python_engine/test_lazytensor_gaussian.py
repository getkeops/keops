# Test for gaussian kernel operation using LazyTensors.

import time

import math
import torch
from pykeops.torch import LazyTensor
from link_compile import hack_eval_lazytensor

M, N, D, DV = 10000, 10000, 3, 1

dtype = torch.float32

sum_scheme = "auto"
c_dtype_acc = "auto"

device_id = "cuda" if torch.cuda.is_available() else "cpu"
do_warmup = False

x = torch.rand(M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)

def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y) ** 2).sum(dim=2)
    Kxy = (-Dxy).exp()
    if backend=="keops_new":
        tmp = Kxy.__matmul__(b, call=False)
        out = hack_eval_lazytensor(tmp, force_recompile=True, sum_scheme=sum_scheme, c_dtype_acc=c_dtype_acc)
    else:
        out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    #print("out:",out.flatten()[:10])
    return out


backends = ["keops_new", "torch"]  # "keops"

out = []
for backend in backends:
    if do_warmup:
        fun(
            x[: min(M, 100), :, :], y[:, : min(N, 100), :], b[: min(N, 100), :], backend
        )
        fun(
            x[: min(M, 100), :, :], y[:, : min(N, 100), :], b[: min(N, 100), :], backend
        )
    start = time.time()
    out.append(fun(x, y, b, backend).squeeze())
    end = time.time()
    print("time for " + backend + ":", end - start)

if len(out) > 1:
    print("relative error:", (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item())

