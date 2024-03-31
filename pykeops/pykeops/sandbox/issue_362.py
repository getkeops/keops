import torch
from pykeops.torch import LazyTensor

A, B, C, D = 32, 8, 16, 400
x = torch.randn(A, B, 1, D).unsqueeze(2).cuda()
w = torch.randn(A, C, D, D).unsqueeze(1).cuda()

# x.shape: (A, B, 1, 1, D)
# w.shape: (A, 1, C, D, D)

res_torch = (x*w).sum(axis=-1).sum(axis=1)
print(res_torch.shape)

xi = LazyTensor(x.view(A, B,   1, D))
wi = LazyTensor(w.view(A, 1, C*D, D))

res_keops = (xi | wi).sum(axis=1).view(A,C,D)
print(res_keops.shape)

print(torch.norm((res_keops-res_torch)/res_torch))
print(torch.max((res_keops-res_torch)/res_torch))
print(torch.min((res_keops-res_torch)/res_torch))
print(torch.mean(torch.abs((res_keops-res_torch)/res_torch)))
