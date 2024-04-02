import torch
from pykeops.torch import LazyTensor
from time import time

A, B, C, D = 32, 8, 16, 400
x = torch.randn(A, B, 1, D).unsqueeze(2).cuda()
w = torch.randn(A, C, D, D).unsqueeze(1).cuda()

# x.shape: (A, B, 1, 1, D)
# w.shape: (A, 1, C, D, D)

start = time()
res_torch_0 = torch.einsum("abde,ace->abd", w.view(A, C, D, D), x.view(A,B,D))
end = time()
print("time for torch 0:", end-start)

start = time()
res_torch = (x*w).sum(axis=-1).sum(axis=1)
end = time()
print("time for torch:", end-start)

print(torch.norm(res_torch_0-res_torch)/torch.norm(res_torch))

start = time()
xi = LazyTensor(x.view(A, B,   1, D))
wi = LazyTensor(w.view(A, 1, C*D, D))
res_keops = (xi | wi).sum(axis=1).view(A,C,D)
print((xi | wi).sum(axis=1).shape)
end = time()
print("time for keops:", end-start)

print(torch.norm(res_keops-res_torch)/torch.norm(res_torch))

start = time()
xp = x.permute(0,2,3,4,1)[...,None,:].contiguous()
wp = w.permute(0,2,3,4,1)[...,None,:].contiguous()
end1 = time()
# xp.shape: (A, 1, 1, D, 1, B)
# wp.shape: (A, C, D, D, 1, 1)

xi = LazyTensor(xp)
wi = LazyTensor(wp)

res_keops_alt = (xi*wi).sum(axis=-1).sum(axis=3).view(A,C,D)
end = time()
print("time for keops alt:", end-start, "(", end1-start, "for permute)")

print(torch.norm(res_keops_alt-res_torch)/torch.norm(res_torch))
