import time

import math
import torch
from pykeops.torch import LazyTensor, Genred
from keopscore.formulas import *
from functools import reduce

M, N, D = 200, 200, 50

test_grad = True

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

do_warmup = True

x = torch.rand(M, D, requires_grad=test_grad, device=device_id)
y = torch.rand(N, D, device=device_id)
gamma = torch.tensor(0.1, device=device_id)

##################################
# SoftDTW operation in pytorch
##################################


def softmin(args, gamma):
    minargs = reduce(lambda x, y: torch.min(x, y), args)
    if gamma > 0:
        minargs -= gamma * sum(((minargs - arg) / gamma).exp() for arg in args).log()
    return minargs


def SoftDTW_torch(x, y, gamma):
    n, m = x.shape[1], y.shape[1]
    x, y = x[:, None, :], y[None, :, :]
    rjm1 = [torch.tensor(torch.inf, device=device_id) for _ in range(n + 1)]
    rjm1[0] = torch.tensor(0.0, device=device_id)
    torchinf = torch.tensor(torch.inf, device=device_id)
    for j in range(1, m + 1):
        rim1j = torchinf
        for i in range(1, n + 1):
            rij = (x[:, :, i - 1] - y[:, :, j - 1]) ** 2 + softmin(
                (rjm1[i], rjm1[i - 1], rim1j), gamma
            )
            rjm1[i - 1] = rim1j
            rim1j = rij
        rjm1[i] = rij
    return rij


#########################################
# reduction function with torch and keops
#########################################


def fun_torch(x, y, gamma):
    Sxy = SoftDTW_torch(x, y, gamma)
    Kxy = (-Sxy).exp()
    return Kxy.sum(dim=1)


def fun_keops(x, y, gamma):
    n, m = x.shape[1], y.shape[1]
    formula = "Exp(-SoftDTW_SqDist(x,y,gamma))"
    aliases = [f"x=Vi({n})", f"y=Vj({m})", "gamma=Pm(1)"]
    Kxy = Genred(formula, aliases, reduction_op="Sum", axis=1)
    return Kxy(x, y, gamma.view((1, 1)))


def fun_lazytensor(x, y, gamma):
    x = LazyTensor(x[:, None, :])
    y = LazyTensor(y[None, :, :])
    sdtw = x.softdtw_sqdist(y, gamma)
    K = (-sdtw).exp()
    return K.sum(axis=1)


##################################
# test
##################################

# funs = (fun_torch, fun_keops, fun_lazytensor)
funs = (fun_torch, fun_lazytensor)
out = []
for fun in funs:
    print("**************************")
    print("Testing " + fun.__name__)
    if do_warmup:
        fun(x[:100, :], y[:100, :], gamma)
        fun(x[:100, :], y[:100, :], gamma)
    start = time.time()
    out.append(fun(x, y, gamma).squeeze())
    end = time.time()
    print("time for " + fun.__name__ + ":", end - start)

print("******")

if len(out) > 1:
    for k in range(1, len(out)):
        print(
            f"relative error {funs[k].__name__} vs {funs[0].__name__}:",
            (torch.norm(out[0] - out[k]) / torch.norm(out[0])).item(),
        )


if test_grad:
    print("Testing grads")
    out_g = []
    for k, fun in enumerate(funs):
        print("**************************")
        print(f"Testing {fun.__name__}")
        start = time.time()
        out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [x])[0])
        end = time.time()
        print("time for " + fun.__name__ + " (grad):", end - start)

    if len(out_g) > 1:
        for k in range(1, len(out)):
            print(
                f"relative error grad {funs[k].__name__} vs {funs[0].__name__}:",
                (torch.norm(out_g[0] - out_g[k]) / torch.norm(out_g[0])).item(),
            )
