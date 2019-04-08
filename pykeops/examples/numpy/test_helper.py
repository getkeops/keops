import numpy as np
from pykeops import keops_formula as kf

# numpy
x = np.random.rand(2,1,1).astype('float32')
y = np.random.rand(1,2,1).astype('float32')
b = np.random.rand(1,2,1).astype('float32')
sigmas = [4.1,2.3,1.05,0.5]

dxy2 = np.sum(abs(x-y)**.5,axis=2)[..., np.newaxis]
Kxy = 0
for s in sigmas:
    Kxy += np.exp(-dxy2/s**2)
res = (Kxy*b).sum(axis=1)

# keops from numpy
x, y, b = kf(x), kf(y), kf(b)

dxy2 = kf.sum(abs(x-y)**.5,axis=2)
Kxy = 0
for s in sigmas:
    Kxy += kf.exp(-dxy2/s**2)
kres = (Kxy*b).sum(axis=1)

# error
print(np.linalg.norm(kres-res))




## torch
#import torch
#x = torch.rand(5,1,3)
#y = torch.rand(1,10,3)
#b = torch.rand(1,10,2)
#sigmas = [4.1,2.3,1.05,0.5]

#dxy2 = torch.sum(abs(x-y)**.5,dim=2)
#Kxy = 0
#for s in sigmas:
    #Kxy += torch.exp(-dxy2/s**2).reshape(5,10,1)
#res = (Kxy*b).sum(dim=1)


## keops from torch
#x, y, b = kf(x), kf(y), kf(b)

#dxy2 = kf.sum(abs(x-y)**.5,dim=2)
#Kxy = 0
#for s in sigmas:
    #Kxy += kf.exp(-dxy2/s**2)
#kres = (Kxy*b).sum(dim=1)

## error
#print(np.linalg.norm(kres-res))




        
        
