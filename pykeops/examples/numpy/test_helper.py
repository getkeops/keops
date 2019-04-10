import numpy as np
from pykeops import keops_formula as kf
from pykeops.numpy.utils import WarmUpGpu

import time

# We use keops formula helper to compute a reduction
# directly from data and using a syntax as close as possible to
# its numpy counterpart

# data
nx, ny = 5000, 10000
x = np.random.rand(nx,1,3)
y = np.random.rand(1,ny,3)
b = np.random.rand(1,ny,4)
sigmas = [4.1,2.3,1.05,0.5]

# first with numpy
start = time.time()
dxy2 = np.sum(abs(x-y)**.5,axis=2)[..., np.newaxis]
Kxy = 0
for s in sigmas:
    Kxy += np.exp(-dxy2/s**2)
res = (Kxy-b)[:,:,1:3].min(axis=1)
print("Timing (Numpy implementation): ",round(time.time()-start,5),"s")
print("numpy output : ",res)

# same with keops
WarmUpGpu()
start = time.time()
x, y, b = kf(x), kf(y), kf(b)
dxy2 = kf.sum(abs(x-y)**.5,axis=2)
Kxy = 0
for s in sigmas:
    Kxy += kf.exp(-dxy2/s**2)
kres = (Kxy-b)[:,:,1:3].min(axis=1)
print("Timing (KeOps implementation): ",round(time.time()-start,5),"s")
print("keops output : ",kres)

# error
print("norm of the difference : ",np.linalg.norm(kres-res))






        
        
