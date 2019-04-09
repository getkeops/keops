import numpy as np
from pykeops import Vi, Vj, Pm


# data
x = np.random.rand(5,3)
y = np.random.rand(10,3)
b = np.random.rand(10,4)
sigma = np.array([4.1, 3.2])

# numpy
dxy2 = np.sum(abs(x[:,None,:]-y[None,:,:])**.5,axis=2)[..., np.newaxis]
Kxy = np.exp(-dxy2/sigma[0]**2) + np.exp(-dxy2/sigma[1]**2)
res = (Kxy*b).sum(axis=1)
print("res=",res)

# keops
X, Y, B, S = Vi(x), Vj(y), Vj(b), Pm(sigma)
dxy2 = (abs(X-Y)**.5).sum()
Kxy = (-dxy2/sigma[0]**2).exp() + (-dxy2/sigma[1]**2).exp()
kres = (Kxy*B).sum_reduction(axis=1)
print("kres=",kres)

# error
print(np.linalg.norm(kres-res))





        
        
