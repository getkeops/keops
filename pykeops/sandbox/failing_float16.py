import torch
from pykeops.torch import LazyTensor as KEOLazyTensor

device = torch.device("cuda:0")
half = torch.float16 # works in float32
par = torch.tensor([[0.8]], requires_grad= True, dtype=half, device=device)
xtrain = torch.randn(1000, 3, device=device, dtype=half)
x1 = xtrain / par
y = torch.randn(1000, 1, dtype=half, device=device) / 31.


# torch
x1_ = x1[..., :, None, :]
x2_ = x1[..., None, :, :]
distance = ((x1_ - x2_) ** 2).sum(-1).sqrt()
exp_component = (-distance).exp()
res_torch = (exp_component @ y)
print(res_torch.requires_grad)
loss = (res_torch**2).sum()
print(torch.autograd.grad(loss, [par], retain_graph=True))



# keops
x1_ = KEOLazyTensor(x1[..., :, None, :])
x2_ = KEOLazyTensor(x1[..., None, :, :])
distance = ((x1_ - x2_) ** 2).sum(-1).sqrt()
exp_component = (-distance).exp()
res_keops = (exp_component @ y)
print(res_keops.requires_grad)
loss = (res_keops**2).sum()
print(torch.autograd.grad(loss, [par]))
