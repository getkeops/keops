import numpy as np
from pykeops import keops_formula as kf
from pykeops import Vi, Vj, Pm


# Here we use keops formula helper to compute a reduction
# directly from data and using Vi, Vj, Pm helpers

# data
x = np.random.rand(5,3)
y = np.random.rand(10,3)
b = np.random.rand(10,4)
sigma = np.array([4.1, 3.2])

# with numpy
dxy2 = np.sum(abs(x[:,None,:]-y[None,:,:])**.5,axis=2)[..., np.newaxis]
Kxy = np.exp(-dxy2/sigma[0]**2) + np.exp(-dxy2/sigma[1]**2)
res = (Kxy*b).sum(axis=1)
print("numpy output : ",res)

# same with keops
x, y, b, sigma = Vi(x), Vj(y), Vj(b), Pm(sigma)
dxy2 = (abs(x-y)**.5).sum()
Kxy = kf.exp(-dxy2/sigma[0]**2) + kf.exp(-dxy2/sigma[1]**2)
kres = (Kxy*b).sum(axis=1)
print("keops output : ",kres)

# error
print("norm of the difference : ",np.linalg.norm(kres-res))






        
        
