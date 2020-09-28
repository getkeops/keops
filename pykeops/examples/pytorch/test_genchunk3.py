# same as test_genchunk.py, but adding batch dimensions

import time

sum_scheme = "block_sum" # "direct_sum" # "kahan_scheme" # 
B1, B2, M, N, D = 3, 2, 1000, 10000, 300
niter = 1

import torch
x = torch.randn(B1, B2, M, 1, D).cuda()
y = torch.randn( 1, B2, 1, N, D).cuda()
b = torch.randn(B1, B2, 1, N, 1).cuda()

from pykeops.torch import LazyTensor
x_i = LazyTensor( x )
y_j = LazyTensor( y )
b_j = LazyTensor( b )
Kb_ij = (-((x_i - y_j)**2).sum(dim=4)/D).exp() * b_j

# dummy calls for better timing
for i in range(1):
	(Kb_ij).sum(dim=3, sum_scheme=sum_scheme, enable_chunks=False)
	(Kb_ij).sum(dim=3, sum_scheme=sum_scheme, enable_chunks=True)

a_i = []
for enable_chunks in [False, True]:
	start = time.time()
	for i in range(niter):
		(Kb_ij).sum(dim=3, sum_scheme=sum_scheme, enable_chunks=enable_chunks)
	end = time.time()
	print("time with enable_chunks=",enable_chunks," : ",(end-start)/niter," s")
	a_i.append( (Kb_ij).sum(dim=3, sum_scheme=sum_scheme, enable_chunks=enable_chunks) )

print("error : ", torch.mean(torch.abs(a_i[0].view(-1)-a_i[1].view(-1))).item())








