from pykeops.torch import LazyTensor
import torch

# a = torch.rand(2, 1000, 5)
a = torch.rand(1000, 5, 2).permute(2, 0, 1)

a.requires_grad = True
b = torch.rand(2, 1000, 5)
c1 = torch.rand(2, 1000, 5)
c2 = torch.rand(2, 1000, 5)

a_i = LazyTensor(a[:, :, None])
b_j = LazyTensor(b[:, None])

dist = a_i.sqdist(b_j)
kernel = dist.exp()
d1 = kernel @ c1
d2 = kernel @ c2

# case 1
d_permute = d1.permute(0, 2, 1)
d_permute.clone().mean().backward()

# case 2
# d_cat = torch.cat([d1,d2],2)
# d_cat.mean().backward()
