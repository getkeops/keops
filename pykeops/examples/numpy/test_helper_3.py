import numpy as np
from pykeops import Vi, Vj, Pm


# Here we use keops formula helper to build symbolic formula

# We define X, Y, B, S as symbolic variables
X, Y, B, S = Vi(0,3), Vj(1,3), Vj(2,4), Pm(3,2)
dxy2 = (abs(X-Y)**.5).sum()
Kxy = (-dxy2/S[0]**2).exp() + (-dxy2/S[1]**2).exp()
keops_op = (Kxy*B).sum(axis=1)

# data
x = np.random.rand(5,3)
y = np.random.rand(10,3)
b = np.random.rand(10,4)
sigma = np.array([4.1, 3.2])

# test the operation
kres = keops_op(x,y,b,sigma)
print("keops output : ",kres)

# same with numpy
dxy2 = np.sum(abs(x[:,None,:]-y[None,:,:])**.5,axis=2)[..., np.newaxis]
Kxy = np.exp(-dxy2/sigma[0]**2) + np.exp(-dxy2/sigma[1]**2)
res = (Kxy*b).sum(axis=1)
print("numpy output : ",res)

# error
print("norm of the difference : ",np.linalg.norm(kres-res))





        
        
