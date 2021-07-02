# Test for gaussian kernel operation using LazyTensors.

import torch
from pykeops.torch import LazyTensor

B, N = 3, 2

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

x = 3+torch.arange(B * N, dtype=dtype, device=device_id).reshape(B, 1, N, 1)
y = 2+torch.arange(B * N, dtype=dtype, device=device_id).reshape(B, N, 1, 1)

def fun(x, y, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    elif backend != "torch":
        raise ValueError("wrong backend")
    out = (x * y).sum(dim=1)
    print("out", backend, ":")
    print(out)
    return out


backends = ["torch", "keops"]

out = []
for backend in backends:
    out.append(fun(x, y, backend).squeeze())

if len(out) > 1:
    print("relative error:", (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item())
