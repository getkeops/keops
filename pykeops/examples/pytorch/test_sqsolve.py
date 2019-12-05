
# Example of use of the "sqsolve" operation on LazyTensors
# Given M random points x_i, N random points y_j, N random values b_j,
# and a scale parameter sigma,
# we solve the matrix system A^T A u = b
# where A is the M*N kernel matrix A_ij = exp(-|x_i-y_j|^2/sigma^2)
# We also test the automatic differentiation of this operation

import torch
from pykeops.torch import LazyTensor

# Set up inputs
M, N, d = 10, 5, 2
x, y, b = torch.rand([M, d],requires_grad=True), torch.rand([N, d],requires_grad=True), torch.randn([N, 1],requires_grad=True)
sigma = 0.25

# Construct kernel matrix
x_i = LazyTensor(x[:, None, :])  # (M, 1, 2)
y_j = LazyTensor(y[None, :, :])  # (1, N, 2)
b_j = LazyTensor(b[None, :, :])  # (1, N, 1)
D_ij = ((x_i - y_j) ** 2).sum(-1)   # (M, N): squared distances
sqrt_K = (-D_ij/sigma**2).exp()

res_keops = sqrt_K.sqsolve(b_j)
loss = (res_keops**2).sum()
loss.backward()

# same with Pytorch

xc, yc, bc = torch.rand([M, d],requires_grad=True), torch.rand([N, d],requires_grad=True), torch.randn([N, 1],requires_grad=True)
xc.data, yc.data, bc.data = x.data, y.data, b.data

x_i = xc[:, None, :]  # (M, 1, 2)
y_j = yc[None, :, :]  # (1, N, 2)
D_ij = ((x_i - y_j) ** 2).sum(-1)   # (M, N): squared distances
sqrt_K = (-D_ij/sigma**2).exp()
K = sqrt_K.t() @ sqrt_K

res_torch = torch.solve(bc,K)[0]
loss = (res_torch**2).sum()
loss.backward()

errf = (torch.norm(res_keops-res_torch)/torch.norm(res_torch)).item()
errgx = (torch.norm(x.grad-xc.grad)/torch.norm(xc.grad)).item()
errgy = (torch.norm(y.grad-yc.grad)/torch.norm(yc.grad)).item()
errgb = (torch.norm(b.grad-bc.grad)/torch.norm(bc.grad)).item()

print("relative error forward : ", errf)
print("relative error grad x : ", errgx)
print("relative error grad y : ", errgy)
print("relative error grad b : ", errgb)
