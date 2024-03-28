from time import time
import torch
from pykeops.torch import LazyTensor, Genred

def timeit(fun,*args,niter=10):
  start = time()
  for i in range(niter):
    res = fun(*args)
    if "cuda" in device:
      torch.cuda.synchronize()
  end = time()
  print(f"mean time for call to {fun.__name__} : ", (end-start)/niter)
  return res

device = "cuda" if torch.cuda.is_available() else "cpu"

I, J, K = 500, 500, 500
test_torch = I*J*K<1e10
a = torch.rand(I, K, device=device)
b = torch.rand(J, K, device=device)

def maxplus_torch(a,b):
  return torch.max(a[:,None,:]+b[None,:,:],axis=2)[0]

def maxplus_keops(a,b):
  I, K = a.shape
  J, K = b.shape
  a_ = LazyTensor(a.view(I, 1, 1, K, 1))
  b_ = LazyTensor(b.view(1, J, 1, K, 1))
  return (a_ + b_).max(dim=3).reshape((I,J)) # (I, J)

def maxplus_keops_2(a,b):
  I, K = a.shape
  J, K = b.shape
  a_ = LazyTensor(a.view(I, 1, 1, K))
  b_ = LazyTensor(b.view(1, J, 1, K))
  return (a_ + b_).max(dim=3).sum(dim=2).reshape((I,J)) # (I, J)

fun_maxplus_keops_3 = Genred("Max(a+b)",[f"a=Vj(0,{K})",f"b=Vi(1,{K})"],axis=1)
def maxplus_keops_3(a,b):
  I, K = a.shape
  J, K = b.shape
  a_ = a.view(I, 1, 1, K)
  b_ = b.view(1, J, 1, K)
  return fun_maxplus_keops_3(a_,b_).reshape((I,J)) # (I, J)

if test_torch:
  res_torch = timeit(maxplus_torch,a,b)
  print()

res_keops = timeit(maxplus_keops,a,b)

if test_torch:
    res_ref = res_torch
    print("error for res_keops:",torch.norm(res_keops-res_ref)/torch.norm(res_ref))
else:
    res_ref = res_keops

print()

res_keops_2 = timeit(maxplus_keops_2,a,b)
print("error for res_keops_2:",torch.norm(res_keops_2-res_ref)/torch.norm(res_ref))

print()

res_keops_3 = timeit(maxplus_keops_3,a,b)
print("error for res_keops_3:",torch.norm(res_keops_3-res_ref)/torch.norm(res_ref))

#import matplotlib.pyplot as plt
#plt.figure(figsize=(8,8))
#plt.imshow((res_keops-res_keops_2).cpu(), cmap='gray')
#plt.show()
