# code from Benjamin, issue #57

import numpy as np
import torch
from pykeops.torch import Genred
import timeit

import matplotlib.pyplot as plt
D_list = np.arange(0,3000,100)
times_list = np.zeros(len(D_list))
times_t_list = np.zeros(len(D_list))
i= -1

for D in D_list:
    i += 1
    d2 = D*D
    a = torch.randn(10000, D, requires_grad=False, dtype=torch.float32).to('cuda')
    c = torch.randn(10000, D, requires_grad=False, dtype=torch.float32).to('cuda')
    v = torch.randn(10000, 2, requires_grad=False, dtype=torch.float32).to('cuda')

    formula = 'Exp((X|Y)/IntCst(' + str(d2) + ')) * v'
    aliases = [
        'X = Vi(%d)' % (a.shape[1]),
        'Y = Vj(%d)' % (c.shape[1]),
        'v = Vj(%d)' % (v.shape[1]),
    ]
    mmv = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype='float32')
    t1 = mmv(a, c, v) # compilation

    times = timeit.repeat("mmv(a, c, v, backend='GPU_1D'); torch.cuda.synchronize()", globals=globals(), number=1, repeat=5)
    times_t = timeit.repeat('((a @ c.T /d2).exp()) @ v; torch.cuda.synchronize()', globals=globals(), number=1, repeat=5)

    times_list[i] = torch.tensor(times).mean().numpy()
    times_t_list[i] = torch.tensor(times_t).mean().numpy()

plt.plot(D_list, times_list, 'o-')
plt.plot(D_list, times_t_list, 'rx-')
plt.savefig("test_keops.png")
