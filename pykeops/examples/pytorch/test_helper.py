import torch
from pykeops import keops_formula as kf


# torch
import torch
x = torch.rand(5,1,3,dtype=torch.float64)
y = torch.rand(1,10,3,dtype=torch.float64)
b = torch.rand(1,10,4,dtype=torch.float64)
sigmas = [4.1,2.3,1.05,0.5]

dxy2 = torch.sum(abs(x-y)**.5,dim=2)
Kxy = 0
for s in sigmas:
    Kxy += torch.exp(-dxy2/s**2)[..., None]
res = (Kxy*b)[:,:,1:3].sum(dim=1)


# keops from torch
x, y, b = kf(x), kf(y), kf(b)

dxy2 = kf.sum(abs(x-y)**.5,dim=2)
Kxy = 0
for s in sigmas:
    Kxy += kf.exp(-dxy2/s**2)
kres = (Kxy*b)[:,:,1:3].sum(dim=1)

# error
print(torch.norm(kres-res).item())




        
        
