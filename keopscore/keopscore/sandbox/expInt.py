from pykeops.numpy import Genred
import numpy as np
import torch
from time import time
import scipy.special as sc

import pykeops



d = 5
n = 15
formula = f"Expn(Sum((x-y)**2), {n})"
#aliases = ["a=Var(0,3,0)", "b=Var(1,3,1)"]
aliases = [f"x=Vi(0,{d})", f"y=Vj(1,{d})"]
myconv = Genred(formula, aliases, reduction_op="Sum", axis=1)
#myconv_np = lambda x,y: np.exp(-np.sum((x[:, np.newaxis,:]-y[np.newaxis,:,:])**2, axis=-1)).sum(axis=1, keepdims=True)
myconv_np = lambda x,y: sc.expn(n,np.sum((x[:, np.newaxis,:]-y[np.newaxis,:,:])**2, axis=-1)).sum(axis=1, keepdims=True)


M, N = 5, 10
x = np.random.randint(1, 4, (M, d)) * 1.
y = np.random.randint(1, 4, (N, d)) * 1.

res = myconv(x, y, backend="GPU")
res_np = myconv_np(x, y)


print(res)
print(res_np)

assert np.allclose(res, res_np)




