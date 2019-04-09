import numpy as np
from pykeops import keops_formula as kf

# We use keops formula helper to compute a reduction
# directly from data and using a syntax as close as possible to
# its numpy counterpart

# data
x = np.random.rand(5,1,3)
y = np.random.rand(1,10,3)
b = np.random.rand(1,10,4)
sigmas = [4.1,2.3,1.05,0.5]

# first with numpy
dxy2 = np.sum(abs(x-y)**.5,axis=2)[..., np.newaxis]
Kxy = 0
for s in sigmas:
    Kxy += np.exp(-dxy2/s**2)
res = (Kxy-b)[:,:,1:3].min(axis=1)
print("numpy output : ",res)

# same with keops
x, y, b = kf(x), kf(y), kf(b)
dxy2 = kf.sum(abs(x-y)**.5,axis=2)
Kxy = 0
for s in sigmas:
    Kxy += kf.exp(-dxy2/s**2)
kres = (Kxy-b)[:,:,1:3].min(axis=1)
print("keops output : ",kres)

# error
print("norm of the difference : ",np.linalg.norm(kres-res))






        
        
