import time

import math
import torch
from pykeops.torch import LazyTensor, Genred
from keopscore.formulas import *
from functools import reduce

M, N, D = 3, 3, 3

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


####################################################################
# helper for SoftDTW operations in keops
####################################################################


def code_softdtw(dtype, out, inputs, n, m, gamma, mode="x,y"):
    if mode == "x,y":
        x, y = inputs

        def diss(i, j):
            return f"""
            rij = {x}[{i}] - {y}[{j}];
            rij *= rij;
            """

    elif mode == "Delta":
        Delta = inputs

        def diss(i, j):
            return f"""
            rij = {Delta}[{j}*{n}+{i}];
            """

    code = f"""
            #define MIN2(a,b) fminf(a,b) //(((a)<(b))?(a):(b))
            #define MIN3(a,b,c) MIN2(MIN2(a,b),c)
            
            {dtype} rjm1[{n}], rim1j, rij, min;
            // j=0, i=0
            {diss("0","0")}
            rim1j = rij;

            // j=0, i=1...n-1
            {use_pragma_unroll()}
            for (int i=1; i<{n}; i++)
            {{
                {diss("i","0")}
                rij += rim1j;
                rjm1[i-1] = rim1j;
                rim1j = rij;
            }}
            rjm1[{n}-1] = rij;

            {use_pragma_unroll()}
            for (int j=1; j<{m}; j++)
            {{
                // j=1...m-1, i=0
                {diss("0","j")}
                rij += rjm1[0];
                rim1j = rij;

                {use_pragma_unroll()}
                for (int i=1; i<{n}; i++)
                {{
                    // j=1...m-1, i=1...n-1
                    {diss("i","j")}
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


####################################################################
# SoftDTW operation in keops for squared difference dissimilarity
####################################################################

from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.code_gen_utils import (
    c_variable,
    pointer,
    c_array,
    c_for_loop,
    c_zero_float,
)


class SoftDTW_L2(Operation):
    string_id = "SoftDTW_L2"

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
        return code_softdtw(dtype, out, (x, y), n, m, gamma, mode="x,y")

    def DiffT(self, v, gradin):
        KeOps_Error("autograd for SoftDTW_L2 operation not yet implemented.")
        pass


import builtins

builtins.SoftDTW_L2 = SoftDTW_L2


####################################################################
# SoftDTW operation in keops (generic, from dissimilarity matrix)
####################################################################

from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.code_gen_utils import (
    c_variable,
    pointer,
    c_array,
    c_for_loop,
    c_zero_float,
)
from keopscore.utils.code_gen_utils import use_pragma_unroll


class SumOverRows(Operation):
    string_id = "SumOverRows"

    def __init__(self, x, n, m, params=()):
        # x is vector of size n*m, interpreted as matrix of shape (n,m)
        super().__init__(x, params=(m, n))
        self.n = n
        self.m = m
        self.dim = m

    def Op(self, out, table, x):
        n, m = self.n, self.m
        code = f"""
            {use_pragma_unroll()}
            for (int j=0; j<{m}; j++)
            {{
                {out}[j] = 0.0;
                {use_pragma_unroll()}
                for (int i=0; i<{n}; i++)
                {{
                    {out}[j] += {x}[j*{n}+i];
                    printf("i,j,out[j] = %d,%d,%f\\n", i,j,out[j]);
                }}
            }}
                """
        return code


class SumOverCols(Operation):
    string_id = "SumOverCols"

    def __init__(self, x, n, m, params=()):
        # x is vector of size n*m, interpreted as matrix of shape (n,m)
        super().__init__(x, params=(m, n))
        self.n = n
        self.m = m
        self.dim = n

    def Op(self, out, table, x):
        n, m = self.n, self.m
        code = f"""
            {use_pragma_unroll()}
            for (int i=0; i<{n}; i++)
                {out}[i] = 0.0;
            {use_pragma_unroll()}
            for (int j=0; j<{m}; j++)
            {{
                {use_pragma_unroll()}
                for (int i=0; i<{n}; i++)
                {{
                    {out}[i] += {x}[j*{n}+i];
                    printf("i,j,out[i] = %d,%d,%f\\n", i,j,out[i]);
                }}
            }}
                """
        return code


class DifferenceMatrix(Operation):
    string_id = "DifferenceMatrix"

    def __init__(self, x, y, params=()):
        # x is vector of size n, y is vector of size m
        super().__init__(x, y, params=())
        self.n = x.dim
        self.m = y.dim
        self.dim = self.n * self.m

    def Op(self, out, table, x, y):
        n, m = self.n, self.m
        code = f"""
            {use_pragma_unroll()}
            for (int j=0; j<{m}; j++)
            {{
                {use_pragma_unroll()}
                for (int i=0; i<{n}; i++)
                {{
                    {out}[j*{n}+i] = {x}[i]-{y}[j];
                }}
            }}
                """

        return code

    def DiffT(self, v, gradin):
        x, y = self.children
        n, m = self.n, self.m
        gradx = SumOverCols(gradin, n, m)
        grady = -SumOverRows(gradin, n, m)
        return x.DiffT(v, gradx) + y.DiffT(v, grady)


import builtins

builtins.DifferenceMatrix = DifferenceMatrix


class SoftDTW(Operation):
    string_id = "SoftDTW"

    def __init__(self, Delta, gamma, input_shape, params=()):
        # Delta is vector of size n*m, interpreted as matrix (n,m), gamma is scalar,
        # output is scalar
        if gamma.dim != 1:
            KeOps_Error("input gamma should be scalar")
        n, m = input_shape
        if n * m != Delta.dim:
            KeOps_Error("inputs dimensions n,m should match size of Delta")
        super().__init__(Delta, gamma, params=(input_shape,))
        self.input_shape = input_shape
        self.dim = 1

    def Op(self, out, table, Delta, gamma):
        dtype = Delta.dtype
        n, m = self.input_shape
        return code_softdtw(dtype, out, Delta, n, m, gamma, mode="Delta")

    def DiffT(self, v, gradin):
        Delta, gamma = self.children
        if v in gamma.Vars_:
            KeOps_Error("autograd wrt gamma in SoftDTW operation not implemented.")
        return Delta.DiffT(v, GradSoftDTW(Delta, gamma, self.input_shape))


import builtins

builtins.SoftDTW = SoftDTW


class GradSoftDTW(Operation):
    string_id = "GradSoftDTW"

    def __init__(self, Delta, gamma, input_shape, params=()):
        # Delta is vector of size n*m, interpreted as matrix (n,m), gamma is scalar,
        # output is scalar
        if gamma.dim != 1:
            KeOps_Error("input gamma should be scalar")
        n, m = input_shape
        if n * m != Delta.dim:
            KeOps_Error("inputs dimensions n,m should match size of Delta")
        super().__init__(Delta, gamma, params=(input_shape,))
        self.n = n
        self.m = m
        self.dim = 1

    def Op(self, out, table, Delta, gamma):
        dtype = Delta.dtype
        n, m = self.n, self.m
        code = f"""
            #define MIN2(a,b) fminf(a,b) //(((a)<(b))?(a):(b))
            #define MIN3(a,b,c) MIN2(MIN2(a,b),c)
            
            {dtype} r[{n*m}], min, a, b, c;

            // Forward pass to fill in r matrix

            // j=0, i=0
            r[0] = {Delta}[0];

            // j=0, i=1...n-1
            {use_pragma_unroll()}
            for (int i=1; i<{n}; i++)
                r[i] = {Delta}[i] + r[i-1];

            {use_pragma_unroll()}
            for (int j=1; j<{m}; j++)
            {{
                // j=1...m-1, i=0
                r[j*{n}] = {Delta}[j*{n}];

                {use_pragma_unroll()}
                for (int i=1; i<{n}; i++)
                {{
                    // j=1...m-1, i=1...n-1
                    r[j*{n}+i] = {Delta}[j*{n}+i];
                    min = MIN3(r[(j-1)*{n}+i-1],r[(j-1)*{n}+i],r[j*{n}+i-1]);
                    r[j*{n}+i] += min - {gamma}[0] * log( exp((min-r[(j-1)*{n}+i-1])/{gamma}[0]) + exp((min-r[j*{n}+i-1])/{gamma}[0]) + exp((min-r[(j-1)*{n}+i])/{gamma}[0]) );
                }}
            }}

            printf("end of forward = %f\\n", r[j*{n}+i]);

            // backward pass

            // j=m-1, i=n-1
            out[{m*n-1}] = 1.0;
            
            // j=m-1, i=n-2..0
            {use_pragma_unroll()}
            for (int i={n-2}; i>=0; i--)
            {{
                a = exp((r[{(m-1)*n}+i+1]-r[{(m-1)*n}+i]-{Delta}[{(m-1)*n}+i+1])/{gamma}[0]);
                out[{(m-1)*n}+i] = a * out[{(m-1)*n}+i+1];
            }}

            {use_pragma_unroll()}
            for (int j={m-2}; j>=0; j--)
            {{
                // j=m-2..0, i=n-1
                b = exp((r[(j+1)*{n}+{n-1}]-r[j*{n}+{n-1}]-{Delta}[(j+1)*{n}+{n-1}])/{gamma}[0]);
                out[j*{n}+{n-1}] = b * out[(j+1)*{n}+{n-1}];

                {use_pragma_unroll()}
                for (int i={n-2}; i>=0; i--)
                {{
                    // j=m-2..0, i=n-2..0
                    a = exp((r[j*{n}+i+1]-r[j*{n}+i]-{Delta}[j*{n}+i+1])/{gamma}[0]);
                    b = exp((r[(j+1)*{n}+i]-r[j*{n}+i]-{Delta}[(j+1)*{n}+i])/{gamma}[0]);
                    c = exp((r[(j+1)*{n}+i+1]-r[j*{n}+i]-{Delta}[(j+1)*{n}+i+1])/{gamma}[0]);
                    out[j*{n}+i] = a * out[j*{n}+i+1] + b * out[(j+1)*{n}+i] + c * out[(j+1)*{n}+i+1];

                    printf("i,j,out[i,j] = %d,%d,%f\\n", i,j,out[j*{n}+i]);
                }}
            }}
                """

        return code

    def DiffT(self, v, gradin):
        KeOps_Error("autograd for GradSoftDTW operation not yet implemented.")
        pass


import builtins

builtins.GradSoftDTW = GradSoftDTW

#########################################
# reduction function with torch and keops
#########################################


def fun_torch(x, y, gamma):
    Sxy = SoftDTW_torch(x, y, gamma)
    Kxy = (-Sxy).exp()
    return Kxy.sum(dim=1)


def fun_keops(x, y, gamma):
    n, m = x.shape[1], y.shape[1]
    formula = "Exp(-SoftDTW_L2(x,y,gamma))"
    aliases = [f"x=Vi({n})", f"y=Vj({m})", "gamma=Pm(1)"]
    Kxy = Genred(formula, aliases, reduction_op="Sum", axis=1)
    return Kxy(x, y, gamma.view((1, 1)))


def fun_lazytensor(x, y, gamma):
    x = LazyTensor(x[:, None, :])
    y = LazyTensor(y[None, :, :])
    sdtw = x.softdtw_l2(y, gamma)
    K = (-sdtw).exp()
    return K.sum(axis=1)


def fun_lazytensor_diffmatrix(x, y, gamma):
    x = LazyTensor(x[:, None, :])
    y = LazyTensor(y[None, :, :])
    dist_l2 = x.difference_matrix(y) ** 2
    sdtw = dist_l2.softdtw(gamma, input_shape=(x.ndim, y.ndim))
    K = (-sdtw).exp()
    return K.sum(axis=1)


##################################
# test
##################################

# funs = (fun_torch, fun_keops, fun_lazytensor, fun_lazytensor_diffmatrix)
funs = (fun_torch, fun_lazytensor_diffmatrix)
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

        print(out_g[-1])
        input()

        print("time for " + fun.__name__ + " (grad):", end - start)

    if len(out_g) > 1:
        for k in range(1, len(out)):
            print(
                f"relative error grad {funs[k].__name__} vs {funs[0].__name__}:",
                (torch.norm(out_g[0] - out_g[k]) / torch.norm(out_g[0])).item(),
            )
