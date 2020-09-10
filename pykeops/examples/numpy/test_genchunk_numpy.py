
import time

sum_scheme = "block_sum" # "direct_sum" # "kahan_scheme" # 
B1, B2, M, N, D = 3, 2, 1000, 10000, 300
niter = 1

import numpy as np
x = np.random.randn(B1, B2, M, 1, D).astype("float32")
y = np.random.randn( 1, B2, 1, N, D).astype("float32")
b = np.random.randn(B1, B2, 1, N, 1).astype("float32")

from pykeops.numpy import LazyTensor
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

print("error : ", np.mean(np.abs(a_i[0]-a_i[1])))








