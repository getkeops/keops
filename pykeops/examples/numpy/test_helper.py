import numpy as np
from pykeops import keops_formula as kf

# numpy
x = np.random.rand(5,1,3)
y = np.random.rand(1,10,3)
b = np.random.rand(1,10,4)
sigmas = [4.1,2.3,1.05,0.5]

dxy2 = np.sum(abs(x-y)**.5,axis=2)[..., np.newaxis]
Kxy = 0
for s in sigmas:
    Kxy += np.exp(-dxy2/s**2)
res = (Kxy-b)[:,:,1:3].min(axis=1)
print("res=",res)

# keops from numpy
X, Y, B = kf(x), kf(y), kf(b)

print(X.dim)
print(Y.dim)
print((X-Y).dim)
dxy2 = kf.sum(abs(X-Y)**.5,axis=2)
Kxy = 0
for s in sigmas:
    Kxy += kf.exp(-dxy2/s**2)
kres = (Kxy-B)[:,:,1:3].min(axis=1)
print("kres=",kres)
print("error : ",np.linalg.norm(kres-res))







        
        
