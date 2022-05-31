from pykeops.torch import LazyTensor
from torch import rand

a = (LazyTensor(rand(size=(2, 1, 99))) | LazyTensor(rand(size=(1, 2, 99)))).argmax(1)
print(a)
