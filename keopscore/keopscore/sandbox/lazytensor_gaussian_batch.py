# Test for gaussian kernel operation using LazyTensors.

import torch
from pykeops.torch import LazyTensor

B1, B2, B3 = 2, 3, 4
D = 3
M, N = 2000, 3000

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

x = torch.rand(B1, 1, B3, M, 1, D, dtype=dtype, device=device_id)
y = torch.rand(1, B2, 1, 1, N, D, dtype=dtype, device=device_id)
p = 1 + torch.arange(B2 * B3, dtype=dtype, device=device_id).reshape(1, B2, B3, 1, 1)


def fun(x, y, p, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
        p = LazyTensor(p.reshape(p.shape + (1,)))
    elif backend != "torch":
        raise ValueError("wrong backend")
    out = ((x - y).sum(dim=5) / p).exp().sum(dim=4)
    # print("out", backend, ":")
    # print(out)
    return out


backends = ["torch", "keops"]

out = []
for backend in backends:
    out.append(fun(x, y, p, backend).squeeze())

if len(out) > 1:
    print("relative error:", (torch.norm(out[0] - out[1]) / torch.norm(out[0])).item())
