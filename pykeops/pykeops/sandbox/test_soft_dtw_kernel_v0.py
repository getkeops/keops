import time

import math
import torch
from pykeops.torch import LazyTensor, Genred
from keopscore.formulas import *
from functools import reduce

# M,N are number of samples xi, yj,
# D is size of each sample
M, N, D = 1000, 1000, 50

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

do_warmup = True

x = torch.rand(M, D, device=device_id)
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


##################################
# SoftDTW operation in keops
##################################

from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.code_gen_utils import (
    c_variable,
    pointer,
    c_array,
    c_for_loop,
    c_zero_float,
)


class SoftDTW(Operation):
    string_id = "SoftDTW"

    def __init__(self, x, y, gamma, params=()):
        # x is vector of size n, y is vector of size m, gamma is scalar,
        # output is scalar
        if gamma.dim != 1:
            KeOps_Error("input gamma should be scalar")
        super().__init__(x, y, gamma)
        self.n = x.dim
        self.m = y.dim
        self.dim = 1

    def Op(self, out, table, x, y, gamma):
        dtype = x.dtype
        n, m = self.n, self.m
        code = f"""
            #define MIN2(a,b) fminf(a,b) //(((a)<(b))?(a):(b))
            #define MIN3(a,b,c) MIN2(MIN2(a,b),c)
            
            {dtype} rjm1[{n}], rim1j, rij, min;

            // j=0, i=0
            rij = {x}[0] - {y}[0];
            rij *= rij;
            rim1j = rij;

            // j=0, i=1...n-1
            #pragma unroll
            for (int i=1; i<{n}; i++)
            {{
                rij = {x}[i] - {y}[0];
                rij *= rij;
                rij += rim1j;
                rjm1[i-1] = rim1j;
                rim1j = rij;
            }}
            rjm1[{n}-1] = rij;

            #pragma unroll
            for (int j=1; j<{m}; j++)
            {{
                // j=1...m-1, i=0
                rij = {x}[0] - {y}[j];
                rij *= rij;
                rij += rjm1[0];
                rim1j = rij;

                #pragma unroll
                for (int i=1; i<{n}; i++)
                {{
                    // j=1...m-1, i=1...n-1
                    rij = {x}[i] - {y}[j];
                    rij *= rij;
                    min = MIN3(rjm1[i-1],rjm1[i],rim1j);
                    rij += min - {gamma}[0] * log( exp((min-rjm1[i-1])/{gamma}[0]) + exp((min-rim1j)/{gamma}[0]) + exp((min-rjm1[i])/{gamma}[0]) );
                    rjm1[i-1] = rim1j;
                    rim1j = rij;
                }}
                rjm1[{n}-1] = rij;
            }}
            {out}[0] = rij;
                """

        return code

    def DiffT(self, v, gradin):
        KeOps_Error("autograd for SoftDTW operation not yet implemented.")
        pass


import builtins

builtins.SoftDTW = SoftDTW


#########################################
# reduction function with torch and keops
#########################################


def fun_torch(x, y, gamma):
    Sxy = SoftDTW_torch(x, y, gamma)
    Kxy = (-Sxy).exp()
    return Kxy.sum(dim=1)


def fun_keops(x, y, gamma):
    n, m = x.shape[1], y.shape[1]
    formula = "Exp(-SoftDTW(x,y,gamma))"
    aliases = [f"x=Vi({n})", f"y=Vj({m})", "gamma=Pm(1)"]
    Kxy = Genred(formula, aliases, reduction_op="Sum", axis=1)
    return Kxy(x, y, gamma.view((1, 1)))


##################################
# test
##################################

funs = (fun_torch, fun_keops)
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
