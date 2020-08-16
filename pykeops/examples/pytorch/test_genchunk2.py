import torch
from pykeops.torch import Genred
import timeit

a = torch.randn(10000, 700, requires_grad=False, dtype=torch.float64)
c = torch.randn(10000, 700, requires_grad=False, dtype=torch.float64)
v = torch.randn(10000, 2, requires_grad=False, dtype=torch.float64)

formula = '(X|Y) * v'
aliases = [
    'X = Vi(%d)' % (a.shape[1]),
    'Y = Vj(%d)' % (c.shape[1]),
    'v = Vi(%d)' % (v.shape[1]),
]
mmv = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype='float64')

print(timeit.repeat("mmv(a, c, v, backend='GPU_1D'); torch.cuda.synchronize()", globals=globals(), number=1, repeat=5))
print(timeit.repeat('(a @ c.T) @ v', globals=globals(), number=1, repeat=5))
