"""
=========
TensorDot
=========

This is a test script to showcase the tensordot syntax.
"""

import numpy as np
import torch

from pykeops.torch import LazyTensor

M, N = 2, 10

#######################################################################################################################
# Matrix multiplication as a special case of Tensordot
# ----------------------------------------------------
#

a = torch.randn(4 * 7, requires_grad=True, dtype=torch.float64)
b = torch.randn(7, requires_grad=True, dtype=torch.float64)
c = a.reshape(4, 7) @ b

#######################################################################################################################
# A single matrix multiplication
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In this case no need to use KeOps: this is a sanity check.

A = LazyTensor(a[None, None, :])
B = LazyTensor(b[None, None, :])
C = A.keops_tensordot(B, (4, 7), (7,), (1,), (0,)).sum_reduction(dim=1)

# print(C, c)
print(
    "Compare the two MatVecMul implementations. All good?",
    torch.allclose(c.flatten(), C.flatten()),
)

xi = torch.randn(4, dtype=torch.float64)
dC = torch.autograd.grad(C, a, xi.reshape(1, 4), retain_graph=True)[0].view(-1)
dc = torch.autograd.grad(c, a, xi, retain_graph=True)[0].view(-1)

# print(dC, dc)
print(
    "Compare the two MatVecMul gradient wrt a implementations. All good?",
    torch.allclose(dc.flatten(), dC.flatten()),
)

dC = torch.autograd.grad(C, b, xi.reshape(1, 4))[0].view(-1)
dc = torch.autograd.grad(c, b, xi)[0].view(-1)

# print(dC, dc)
print(
    "Compare the two MatVecMul gradient wrt b implementations. All good?",
    torch.allclose(dc.flatten(), dC.flatten()),
)

print("-------------------------------")

#######################################################################################################################
# Matrix multiplication with a sum reduction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# That is where KeOps come into play.

a = torch.randn(M, 4 * 7, requires_grad=True, dtype=torch.float64)
b = torch.randn(N, 7, requires_grad=True, dtype=torch.float64)
c = torch.tensordot(a.reshape(M, 4, 7), b.reshape(N, 7), dims=([2], [1])).sum(2)

A = LazyTensor(a[:, None, :])
B = LazyTensor(b[None, :, :])
C = A.keops_tensordot(B, (4, 7), (7,), (1,), (0,)).sum_reduction(dim=1)

# print(C, c)
print(
    "Compare the two MatVecMul with sum implementations. All good ?",
    torch.allclose(c.flatten(), C.flatten()),
)

xi = torch.randn(M, 4, dtype=torch.float64)
dCa = torch.autograd.grad(C, a, xi, retain_graph=True)[0].view(-1)
dca = torch.autograd.grad(c, a, xi, retain_graph=True)[0].view(-1)

# print(dC, dc)
print(
    "Compare the two MatVecMul with sum gradient wrt a implementations. All good ?",
    torch.allclose(dca.flatten(), dCa.flatten()),
)

dCb = torch.autograd.grad(C, b, xi)[0].view(-1)
dcb = torch.autograd.grad(c, b, xi)[0].view(-1)

# print(dC, dc)
print(
    "Compare the two MatVecMul with sum gradient wrt b implementations. All good ?",
    torch.allclose(dcb.flatten(), dCb.flatten()),
)

print("-------------------------------")

#######################################################################################################################
# Matrix-Matrix multiplication as a special case of Tensordot
# -----------------------------------------------------------
#

a = torch.randn(4 * 7, requires_grad=True, dtype=torch.float64)
b = torch.randn(7 * 2, requires_grad=True, dtype=torch.float64)
c = a.reshape(4, 7) @ b.reshape(7, 2)

A = LazyTensor(a[None, None, :])
B = LazyTensor(b[None, None, :])
C = A.keops_tensordot(B, (4, 7), (7, 2), (1,), (0,)).sum_reduction(dim=1)

# print(C, c)
print(
    "Compare the two MatMul implementations. All good?",
    torch.allclose(c.flatten(), C.flatten()),
)

xi = torch.randn(4 * 2, dtype=torch.float64)
dC = torch.autograd.grad(C, a, xi.reshape(1, 4 * 2), retain_graph=True)[0].view(-1)
dc = torch.autograd.grad(c, a, xi.reshape(4, 2), retain_graph=True)[0].view(-1)

# print(dC, dc)
print(
    "Compare the two MatMul gradient wrt a implementations. All good?",
    torch.allclose(dc.flatten(), dC.flatten()),
)

dCb = torch.autograd.grad(C, b, xi.reshape(1, 4 * 2))[0].view(-1)
dcb = torch.autograd.grad(c, b, xi.reshape(4, 2))[0].view(-1)

# print(dCb, dcb)
print(
    "Compare the two MatMul gradient wrt b implementations. All good?",
    torch.allclose(dcb.flatten(), dCb.flatten()),
)

print("-------------------------------")

#######################################################################################################################
# Tensordot in keops (generic case)
# ---------------------------------
#
# A fisrt example
# ^^^^^^^^^^^^^^^
#
# First, let us start with a standard torch implementation. We contract two tensor along a common axis of size 7.
# Then, a reduction is performed alog the dimension of size N.

x = torch.randn(M, 4, 7, 3, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 7, 2, requires_grad=True, dtype=torch.float64)

f_torch = torch.tensordot(x, y, dims=([2], [1]))  # now is shape (M, 4, 3, N, 2)
sum_f_torch2 = f_torch.sum(3)  # ... yielding a result of dimension (M,4*3*2)

# In KeOps, we forgot the first reduction axis (size M and N respectively). We then need to tell the compiler not only
# the contration axis (1 and 0 respectively both of dimension 7) but the shapes (4,7,3) and (7,2) as well,
# keeping in mind that the 2 actual first axis of x and y (reduction axis) are ignored so the result has
# shape (M,4*3*2) or (N, 4*3*2) depending on the chosen reduction axis.

f_keops = LazyTensor(x.reshape(M, 1, 4 * 7 * 3)).keops_tensordot(
    LazyTensor(y.reshape(1, N, 7 * 2)), (4, 7, 3), (7, 2), (1,), (0,)
)
sum_f_keops = f_keops.sum_reduction(dim=1)  # reduction is perform along second axis
# print(sum_f_keops.flatten())                        # ... yielding a result of dimension (M,4*3*2)

print(
    "Compare the two tensordot implementation. All good ?",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten(), rtol=1e-4),
)

########################################################################################################################
# As before, let us check the gradients

e = torch.randn(M, 4 * 3 * 2, dtype=torch.float64)
Ee = e.reshape(M, 4, 3, 2)
grad_keops = (
    torch.autograd.grad(sum_f_keops, x, e, retain_graph=True)[0].squeeze().numpy()
)
grad_torch = (
    torch.autograd.grad(sum_f_torch2, x, Ee, retain_graph=True)[0].squeeze().numpy()
)

# print(grad_keops[0,:,:,:])
# print(grad_torch[0,:,:,:])
print(
    "Check gradient wrt x. All good ?",
    np.allclose(grad_keops.flatten(), grad_torch.flatten()),
)

# tmp = torch.tensordot(Ee,y, dims=([3], [2])).sum(3).detach().numpy()
# print("grad_keops and tmp are the same? ", np.allclose(tmp.flatten(), grad_keops.flatten()))

# print("grad_torch and tmp are the same? ",  np.allclose(grad_torch , np.moveaxis(tmp, [0,1,2,3], [0,1,3,2])))
grad_keops = torch.autograd.grad(sum_f_keops, y, e)[0].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, Ee)[0].numpy()

# print(grad_keops[:1])
# print(grad_torch[:1])
print(
    "Check gradient wrt y. All good ?",
    np.allclose(grad_keops.flatten(), grad_torch.flatten()),
)

print("-------------------------------")

#######################################################################################################################
# A Second example
# ^^^^^^^^^^^^^^^^
#
# Torch version

x = torch.randn(M, 4, 3, 7, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 7, 2, requires_grad=True, dtype=torch.float64)

f_torch = torch.tensordot(x, y, dims=([3], [1]))  # now is shape (M, 4, 3, N, 2)
sum_f_torch2 = f_torch.sum(3)  # ... yielding a result of dimension (M,4,3,2)

#######################################################################################################################
# And corresponding KeOps version

f_keops = LazyTensor(x.reshape(M, 1, 4 * 3 * 7)).keops_tensordot(
    LazyTensor(y.reshape(1, N, 7 * 2)), (4, 3, 7), (7, 2), (2,), (0,)
)
sum_f_keops = f_keops.sum_reduction(dim=1)  # reduction is perform along second axis
# print(sum_f_keops.shape)                        # ... yielding a result of dimension (M,4*3*2)

print(
    "Compare the two tensordot implementation. All good ?",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten(), rtol=1e-4),
)

# checking gradients
e = torch.randn(M, 4 * 3 * 2, dtype=torch.float64)
Ee = e.reshape(M, 4, 3, 2)
grad_keops = (
    torch.autograd.grad(sum_f_keops, x, e, retain_graph=True)[0].squeeze().numpy()
)
grad_torch = (
    torch.autograd.grad(sum_f_torch2, x, Ee, retain_graph=True)[0].squeeze().numpy()
)

# print(grad_keops[0,:,:,:])
# print(grad_torch[0,:,:,:])
print(
    "Compare the two gradient x tensordot implementation. All good ?",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

grad_keops = (
    torch.autograd.grad(sum_f_keops, y, e, retain_graph=True)[0].squeeze().numpy()
)
grad_torch = (
    torch.autograd.grad(sum_f_torch2, y, Ee, retain_graph=True)[0].squeeze().numpy()
)

# print(grad_keops[0,:,:,:])
# print(grad_torch[0,:,:,:])
print(
    "Compare the two gradient y tensordot implementation. All good ?",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

print("------------------------------------------")

#######################################################################################################################
# A Third example
# ^^^^^^^^^^^^^^^^

x = torch.randn(M, 4, 3, 2, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 4, 2, requires_grad=True, dtype=torch.float64)

xshape, yshape = x.shape[1:], y.shape[1:]
f_keops = LazyTensor(x.reshape(M, 1, int(np.array((xshape)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(yshape).prod()))),
    xshape,
    yshape,
    (0, 2),
    (0, 1),
)
sum_f_keops = f_keops.sum_reduction(dim=1)
sum_f_torch2 = torch.tensordot(x, y, dims=([1, 3], [1, 2])).sum(2)
# sum_f_torch2 = torch.tensordot(x, y, dims=([3], [1])).sum(3)

print(
    "Compare the two tensordot implementation. All good ????",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()
print(
    "Compare the two gradient x tensordot implementation. is All good ????",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, e, retain_graph=True)[0].numpy()
print(
    "Compare the two gradient y tensordot implementation. is All good ????",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

print("------------------------------------------")

#######################################################################################################################
# A Fourth example
# ^^^^^^^^^^^^^^^^

x = torch.randn(M, 2, 3, 4, 2, 2, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 2, 4, 5, 3, 2, requires_grad=True, dtype=torch.float64)

xshape, yshape = x.shape[1:], y.shape[1:]
f_keops = LazyTensor(x.reshape(M, 1, int(np.array((xshape)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(yshape).prod()))),
    xshape,
    yshape,
    (0, 1, 4),
    (0, 3, 4),
)
sum_f_keops = f_keops.sum_reduction(dim=1)
sum_f_torch2 = torch.tensordot(x, y, dims=([1, 2, 5], [1, 4, 5])).sum(3)
# sum_f_torch2 = torch.tensordot(x, y, dims=([3], [1])).sum(3)

print(
    "Compare the two tensordot implementation. All good ????!",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()

print(
    "Compare the two gradient x tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, e, retain_graph=True)[0].numpy()
print(
    "Compare the two gradient y tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

print("------------------------------------------")

#######################################################################################################################
# A Fifth example
# ^^^^^^^^^^^^^^^^


x = torch.randn(M, 2, 3, 4, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 2, 4, 5, requires_grad=True, dtype=torch.float64)

xshape, yshape = x.shape[1:], y.shape[1:]
f_keops = LazyTensor(x.reshape(M, 1, int(np.array((xshape)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(yshape).prod()))),
    xshape,
    yshape,
    (2, 0),
    (1, 0),
)
sum_f_keops = f_keops.sum_reduction(dim=1)
sum_f_torch2 = torch.tensordot(x, y, dims=([3, 1], [2, 1])).sum(2)
# sum_f_torch2 = torch.tensordot(x, y, dims=([3], [1])).sum(3)

print(
    "Compare the two tensordot implementation. All good ????!",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()

print(
    "Compare the two gradient x tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, e, retain_graph=True)[0].numpy()
print(
    "Compare the two gradient y tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

print("------------------------------------------")
#######################################################################################################################
# A Sixth example
# ^^^^^^^^^^^^^^^^

x = torch.randn(M, 2, 3, 4, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 4, 2, requires_grad=True, dtype=torch.float64)

xshape, yshape = x.shape[1:], y.shape[1:]
f_keops = LazyTensor(x.reshape(M, 1, int(np.array((xshape)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(yshape).prod()))),
    xshape,
    yshape,
    (2, 0),
    (0, 1),
)
sum_f_keops = f_keops.sum_reduction(dim=1)
sum_f_torch2 = torch.tensordot(x, y, dims=([3, 1], [1, 2])).sum(2)
# sum_f_torch2 = torch.tensordot(x, y, dims=([3], [1])).sum(3)

print(
    "Compare the two tensordot implementation. All good ????",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()

print(
    "Compare the two gradient x tensordot implementation. All good ????",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1))[0].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, e)[0].numpy()
# print(grad_keops)
# print(grad_torch)
print(
    "Compare the two gradient y tensordot implementation. All good ????",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

print("------------------------------------------")

#######################################################################################################################
# A Seventh example
# ^^^^^^^^^^^^^^^^^

x = torch.randn(M, 2, 3, 2, 2, 4, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 2, 4, 2, 3, 2, 3, requires_grad=True, dtype=torch.float64)

xshape, yshape = x.shape[1:], y.shape[1:]
f_keops = LazyTensor(x.reshape(M, 1, int(np.array((xshape)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(yshape).prod()))),
    xshape,
    yshape,
    (4, 0, 2),
    (1, 4, 2),
)
sum_f_keops = f_keops.sum_reduction(dim=1)
sum_f_torch2 = torch.tensordot(x, y, dims=([5, 1, 3], [2, 5, 3])).sum(3)
# sum_f_torch2 = torch.tensordot(x, y, dims=([3], [1])).sum(3)

print(
    "Compare the two tensordot implementation. All good ????!",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()

print(
    "Compare the two gradient x tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, e, retain_graph=True)[0].numpy()
print(
    "Compare the two gradient y tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

print("------------------------------------------")


#######################################################################################################################
#
#
#


def my_tensordort_perm(a, b, dims=None, perm=None):
    # print(torch.tensordot(a, b, dims=dims).sum(3).shape)
    return torch.tensordot(a, b, dims=dims).sum(3).permute(perm)


def invert_permutation_numpy(permutation):
    return np.arange(len(permutation))[np.argsort(permutation)]


x = torch.randn(M, 2, 3, 4, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 2, 4, requires_grad=True, dtype=torch.float64)

dimfa, dimfb = x.shape[1:], y.shape[1:]
contfa, contfb = [3], [2]
keepfa, keepfb = [item - 1 for item in [1, 2, 3] if item not in contfa], [
    item for item in [1, 2] if item not in contfb
]
# contfa, contfb = [2, 3], [1, 2]
n = len(dimfa) + len(dimfb) - 2 * len(contfa)
# perm = [int(i) for i in torch.randperm(n)]
perm = [2, 0, 1]
# perm = [2, 1, 3, 0]
# perm = [1, 0]


perm_torch = (0,) + tuple([(i + 1) for i in invert_permutation_numpy(perm)])
sum_f_torch2 = my_tensordort_perm(
    x, y, dims=(contfa, contfb), perm=perm_torch
)  # 1, 2,3,5,4 -> 1, 5,3,4,2

f_keops = LazyTensor(x.reshape(M, 1, int(np.array((dimfa)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(dimfb).prod()))),
    dimfa,
    dimfb,
    tuple(np.array(contfa) - 1),
    tuple(np.array(contfb) - 1),
    tuple(perm),
)
sum_f_keops = f_keops.sum_reduction(dim=1)

print(
    "Compare the two tensordot implementation. All good ????!!",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
# grad_torch2 = my_tensordort_perm(e, y, dims=([4,2], keepfb), perm=[0,1,2])

grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()

print(
    "Compare the two gradient x tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)
# print("Compare the two gradient x tensordot implementation. All good ????!",
#       np.allclose(grad_torch2.detach().numpy(), grad_torch, rtol=1e-4))
grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, e, retain_graph=True)[0].numpy()

# grad_torch2 = my_tensordort_perm(e, x, dims=([1,3], [1,2]), perm=[0,1,2,3]).permute(perm)
# grad_torch2 = my_tensordort_perm(e, x, dims=([1,3], [1,2]), perm=perm)

print(
    "Compare the two gradient y tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)
# print("Compare the two gradient y tensordot implementation. All good ????!",
#       np.allclose(grad_torch2.detach().numpy(), grad_torch, rtol=1e-4))
print("------------------------------------------")

x = torch.randn(M, 2, 3, 4, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 2, 4, 5, requires_grad=True, dtype=torch.float64)

dimfa, dimfb = x.shape[1:], y.shape[1:]
contfa, contfb = [3], [2]
keepfa, keepfb = [item - 1 for item in [1, 2, 3] if item not in contfa], [
    item for item in [1, 2, 3] if item not in contfb
]
# contfa, contfb = [2, 3], [1, 2]
n = len(dimfa) + len(dimfb) - 2 * len(contfa)
# perm = [int(i) for i in torch.randperm(n)]
perm = [0, 2, 3, 1]
# perm = [2, 1, 3, 0]
# perm = [1, 0]


perm_torch = (0,) + tuple([(i + 1) for i in invert_permutation_numpy(perm)])
sum_f_torch2 = my_tensordort_perm(
    x, y, dims=(contfa, contfb), perm=perm_torch
)  # 1, 2,3,5,4 -> 1, 5,3,4,2

f_keops = LazyTensor(x.reshape(M, 1, int(np.array((dimfa)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(dimfb).prod()))),
    dimfa,
    dimfb,
    tuple(np.array(contfa) - 1),
    tuple(np.array(contfb) - 1),
    tuple(perm),
)
sum_f_keops = f_keops.sum_reduction(dim=1)

print(
    "Compare the two tensordot implementation. All good ????!!",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
# grad_torch2 = my_tensordort_perm(e, y, dims=([4,2], keepfb), perm=[0,1,2,3])

grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()

print(
    "Compare the two gradient x tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)
# print("Compare the two gradient x tensordot implementation. All good ????!",
#       np.allclose(grad_torch2.detach().numpy(), grad_torch, rtol=1e-4))


grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1), retain_graph=True)[0]
grad_torch = torch.autograd.grad(sum_f_torch2, y, e, retain_graph=True)[0]
# grad_torch2 = my_tensordort_perm(e, x, dims=([1,3], [1,2]), perm=perm)

print(
    "Compare the two gradient y tensordot implementation. All good ????!",
    np.allclose(grad_keops.numpy().flatten(), grad_torch.numpy().flatten(), rtol=1e-4),
)
# print("Compare the two gradient y tensordot implementation. All good ????!",
#      np.allclose(grad_torch2.detach().numpy(), grad_torch, rtol=1e-4))
print("------------------------------------------")

x = torch.randn(M, 2, 3, 2, 2, 4, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 2, 4, 2, 3, 2, 3, requires_grad=True, dtype=torch.float64)

dimfa, dimfb = x.shape[1:], y.shape[1:]
contfa, contfb = [5, 1, 3], [2, 5, 3]
n = len(dimfa) + len(dimfb) - 2 * len(contfa)
# perm_id = [int(i) for i in range(n+1)]

# perm = [int(i) for i in torch.randperm(n)]
# perm = [0,2,1,4,3]
perm = [4, 3, 2, 0, 1]
perm_torch = (0,) + tuple([(i + 1) for i in invert_permutation_numpy(perm)])
sum_f_torch2 = my_tensordort_perm(x, y, dims=(contfa, contfb), perm=perm_torch)

# print(sum_f_torch2.shape)

f_keops = LazyTensor(x.reshape(M, 1, int(np.array((dimfa)).prod()))).keops_tensordot(
    LazyTensor(y.reshape(1, N, int(np.array(dimfb).prod()))),
    dimfa,
    dimfb,
    tuple(np.array(contfa) - 1),
    tuple(np.array(contfb) - 1),
    tuple(perm),
)
sum_f_keops = f_keops.sum_reduction(dim=1)

print(
    "Compare the two tensordot implementation. All good ????!!",
    torch.allclose(sum_f_keops.flatten(), sum_f_torch2.flatten()),
)

# checking gradients
e = torch.randn_like(sum_f_torch2)
grad_keops = torch.autograd.grad(sum_f_keops, x, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, x, e, retain_graph=True)[0].numpy()
# grad_torch2 = my_tensordort_perm(e, y, dims=([1,2,3], [1,4,6]), perm=[0,1,2,3,4,5]).permute(3)

print(
    "Compare the two gradient x tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

grad_keops = torch.autograd.grad(sum_f_keops, y, e.reshape(M, -1), retain_graph=True)[
    0
].numpy()
grad_torch = torch.autograd.grad(sum_f_torch2, y, e, retain_graph=True)[0].numpy()
print(
    "Compare the two gradient y tensordot implementation. All good ????!",
    np.allclose(grad_keops.flatten(), grad_torch.flatten(), rtol=1e-4),
)

print("------------------------------------------")


#######################################################################################################################
#  Using gradcheck
#  ---------------
#


# def my_tensordot(x,y):
# f_keops = LazyTensor(x.reshape(M, 1, 4 * 3 * 7)).keops_tensordot(LazyTensor(y.reshape(1, N, 7 * 2)), (4, 3, 7),
# (7, 2), (2,), (0,))
# return f_keops.sum_reduction(dim=1)

# print(torch.autograd.gradcheck(my_tensordot, [x,y]))


def my_tensordot2(x, y):
    xshape, yshape = x.shape[1:], y.shape[1:]
    f_keops = LazyTensor(
        x.reshape(M, 1, int(np.array((xshape)).prod()))
    ).keops_tensordot(
        LazyTensor(y.reshape(1, N, int(np.array(yshape).prod()))),
        xshape,
        yshape,
        (2, 0),  # (2,0,1),
        (0, 1),  # (0,3,2)
    )
    return f_keops.sum_reduction(dim=1)


x = torch.randn(M, 2, 2, 2, requires_grad=True, dtype=torch.float64)
y = torch.randn(N, 2, 2, requires_grad=True, dtype=torch.float64)
print(torch.autograd.gradcheck(my_tensordot2, [x, y], atol=1e-5, rtol=1e-5))
print(torch.autograd.gradgradcheck(my_tensordot2, [x, y], atol=1e-5, rtol=1e-5))
