import torch
from torch.autograd import grad as grad
from pykeops import keops_formula as kf

# We use keops formula helper to compute a reduction
# directly from data and using a syntax as close as possible to
# its pytorch counterpart


# data
import torch
x = torch.rand(5,1,3, requires_grad=True)
y = torch.rand(1,10,3)
b = torch.rand(1,10,4)
e = torch.rand(5,2)
sigmas = [4.1,2.3,1.05,0.5]

# first with pytorch
dxy2 = torch.sum(abs(x-y)**.5,dim=2)
Kxy = 0
for s in sigmas:
    Kxy += torch.exp(-dxy2/s**2)[..., None]
res = (Kxy*b)[:,:,1:3].sum(dim=1)
print("pytorch output : ",res)

# same with keops
X, Y, B = kf(x), kf(y), kf(b)
dxy2 = kf.sum(abs(X-Y)**.5,dim=2)
Kxy = 0
for s in sigmas:
    Kxy += kf.exp(-dxy2/s**2)
kres = (Kxy*B)[:,:,1:3].sum(dim=1)
print("keops output : ",kres)

# error
print("norm of the difference : ",torch.norm(kres-res).item())

# comparing gradients

gres = grad(res,x,e)[0]
print("pytorch grad : ",gres)
gkres = grad(kres,x,e)[0]
print("keops grad : ",gkres)
print("norm of the difference : ",torch.norm(gkres-gres).item())




        
        
