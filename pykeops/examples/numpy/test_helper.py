import numpy as np
from pykeops.numpy import keops_formula as kf

# numpy 
x = np.random.rand(5,1,3)
y = np.random.rand(1,10,3)
res = (np.exp(-(x-y)**2)*y).sum(axis=1)

# keops
x = kf(x)
y = kf(y)
kres = (kf.exp(-(x-y)**2)*y).sum(axis=1)

# error
print(np.linalg.norm(kres-res))



        
        
