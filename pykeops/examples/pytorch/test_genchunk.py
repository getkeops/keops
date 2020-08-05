
D = 300

import torch
x = torch.randn(10, D).cuda()
y = torch.randn(20, D).cuda()
z = torch.randn(20, D).cuda()
b = torch.randn(20).cuda()
K = torch.exp(-((x[:,None,:] - y[None,:,:]*z[None,:,:])**2).sum(dim=2)/D)


from pykeops.torch import LazyTensor
x_i = LazyTensor( x[:,None,:] )
y_j = LazyTensor( y[None,:,:] )
z_j = LazyTensor( z[None,:,:] )
b_j = LazyTensor( b[None,:,None] )
K_ij = (-((x_i - y_j*z_j)**2).sum(dim=2)/D).exp()



# this works :

a = torch.sum(K,dim=1)
a_i = (K_ij).sum(dim=1).view(-1)

print("a=", a)
print("a_i=",a_i)
print("error : ", torch.mean(torch.abs(a_i-a)).item())




# this does not work currently :

a = torch.sum(K*b[None,:],dim=1)
a_i = (K_ij*b_j).sum(dim=1).view(-1)
print("a=", a)
print("a_i=",a_i)
print("error : ", torch.mean(torch.abs(a_i-a)).item())
