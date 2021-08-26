#%%
import numpy as np

x = np.array([[2.5, 1.5, 3.5], [2.5, 1.5, 3.5]])
y = np.array([[3., 0.6, 5.], [3., 0.6, 5.]])
w = np.array([[1., 2., 1.], [3., 1., 4.]])

from pykeops.numpy import LazyTensor as LazyTensor_np

x_i = LazyTensor_np(
    x[:, None, :]
)  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
y_j = LazyTensor_np(
    y[None, :, :]
)  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y

V_ij = (x_i - y_j)
S_ij = V_ij.sum()

S_ij.logsumexp(0)

#%%
a = x_i.exp()
a.sum(1) # idem que exp(x)
a.sum(0)