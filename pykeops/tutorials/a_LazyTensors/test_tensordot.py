
# This test program for tensordot is temporary, because we will change
# tensordot syntax in LazyTensor soon.

import torch
from pykeops.torch import LazyTensor

M, N = 10, 20
x = torch.rand(M, 1, 2*2*2, requires_grad=True)
y = torch.rand(1, N, 2*2)

# tensordot in keops : we input the fictious shapes (2,2,2) and (2,2)
# and the summation axis 2 and 0, keeping in mind that the 2 actual first axis 
# of x and y (reduction axis) are ignored
# so the result has shape (M,N,2*2*2)
f_keops = LazyTensor(x).keops_tensordot(y,(2,2,2),(2,2),(2,),(0,))
print(f_keops.shape)
sum_f_keops = f_keops.sum_reduction(dim=1)
print(sum_f_keops.shape) # now is (M,2*2*2)
print(sum_f_keops[:10])

# same with pytorch : here we use pytorch's tensordot, so we change the shapes of x and y,
# then we must give 4 and 2 as axis of summations, instead of 2 and 0,
# and morevover the result will have shape (M,1,2,2,1,N,2) instead of (M,N,2*2*2), so we
# have to modiify further in order to get (M,N,2*2*2) as before
f_torch = torch.tensordot(x.view(M,1,2,2,2),y.view(1,N,2,2),dims=([4],[2]))
f_torch = f_torch.permute(0,5,2,3,6,1,4).contiguous().view((M,N,8)) # now is (M,N,2*2*2)
sum_f_torch = f_torch.sum(dim=1)
print(sum_f_torch[:10])

print((sum_f_keops-sum_f_torch).abs().max())





