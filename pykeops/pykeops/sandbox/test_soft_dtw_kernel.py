import time

import math
import torch
from pykeops.torch import LazyTensor, Genred
from keopscore.formulas import *

M, N, D = 10000, 10000, 50

test_grad = False

device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

do_warmup = True

x = torch.rand(M, 1, D, requires_grad=test_grad, device=device_id)
y = torch.rand(1, N, D, device=device_id)
gamma = torch.tensor(.1, device=device_id)

backends = []
if M * N * D < 1e8:
    backends.append("torch")
if D<5:
    backends.append("lazytensor_naive")
if D<15:
    backends.append("lazytensor")
#backends.append("keopscore")
backends.append("keops_custom")

def SoftDTW_torch(x, y, gamma):
    n, m = x.shape[2], y.shape[2]
    r = [[None] * m for _ in range(n)]
    for j in range(m):
        for i in range(n):
            dij = (x[:, :, i] - y[:, :, j]) ** 2
            if i == 0 and j == 0:
                r[i][j] = dij
            elif i == 0:
                r[i][j] = dij + r[i][j - 1]
            elif j == 0:
                r[i][j] = dij + r[i - 1][j]
            else:
                r[i][j] = (
                    dij
                    - gamma
                    * sum(
                        (-a / gamma).exp()
                        for a in (r[i - 1][j - 1], r[i - 1][j], r[i][j - 1])
                    ).log()
                )
    return r[n - 1][m - 1]

def SoftDTW_keops_naive(x, y, gamma):
    start = time.time()
    x,y,gamma = (LazyTensor(item) for item in (x,y,gamma))
    n, m = x.shape[2], y.shape[2]
    r = [[None] * m for _ in range(n)]
    for j in range(m):
        for i in range(n):
            dij = (x[:, :, i] - y[:, :, j]) ** 2
            if i == 0 and j == 0:
                r[i][j] = dij
            elif i == 0:
                r[i][j] = dij + r[i][j - 1]
            elif j == 0:
                r[i][j] = dij + r[i - 1][j]
            else:
                r[i][j] = (
                    dij
                    - gamma
                    * sum(
                        (-a / gamma).exp()
                        for a in (r[i - 1][j - 1], r[i - 1][j], r[i][j - 1])
                    ).log()
                )
    end = time.time()
    print("time for building lazytensor:", end-start)
    return r[n - 1][m - 1]

def SoftDTW_keops(x, y, gamma):
    start = time.time()
    x,y,gamma = (LazyTensor(item) for item in (x,y,gamma))
    n, m = x.shape[2], y.shape[2]
    r = [[None] * m for _ in range(n)]
    v = [[None] * m for _ in range(n)]
    aliasind = -1
    for j in range(m):
        for i in range(n):
            dij = (x[:, :, i] - y[:, :, j]) ** 2
            if i == 0 and j == 0:
                r[i][j] = dij
            elif i == 0:
                r[i][j] = dij + v[i][j - 1]
            elif j == 0:
                r[i][j] = dij + v[i - 1][j]
            else:
                r[i][j] = (
                    dij
                    - gamma
                    * sum(
                        (-a / gamma).exp()
                        for a in (v[i - 1][j - 1], v[i - 1][j], v[i][j - 1])
                    ).log()
                )
            v[i][j] = LazyTensor((aliasind, 1, 3))
            v[i][j].symbolic_variables = ()
            aliasind -= 1
    for j in range(m-1,-1,-1):
        for i in range(n-1,-1,-1):
            if not(j==m-1 and i==n-1):
                r[n-1][m-1].formula = f"Factorize_Impl({r[n-1][m-1].formula},{r[i][j].formula},{v[i][j].formula})"
    end = time.time()
    print("time for building lazytensor:", end-start)
    return r[n - 1][m - 1]


mykernel = None

def SoftDTW_keopscore(x,y,gamma):
    start = time.time()
    n,m = x.dim, y.dim
    r = [[None] * m for _ in range(n)]
    v = [[None] * m for _ in range(n)]
    aliasind = -1
    for j in range(m):
        for i in range(n):
            dij = (Elem(x,i) - Elem(y,j)) ** 2
            if i == 0 and j == 0:
                r[i][j] = dij
            elif i == 0:
                r[i][j] = dij + v[i][j - 1]
            elif j == 0:
                r[i][j] = dij + v[i - 1][j]
            else:
                r[i][j] = (
                    dij
                    - gamma
                    * Log(sum(
                        Exp(-a / gamma)
                        for a in (v[i - 1][j - 1], v[i - 1][j], v[i][j - 1])
                    ))
                )
            v[i][j] = Var(aliasind, 1, 3)
            aliasind -= 1
    tmp = []
    for j in range(m-1,-1,-1):
        for i in range(n-1,-1,-1):
            if j==m-1 and i==n-1:
                tmp.append(r[n-1][m-1])
            else:
                tmp.append((r[i][j],v[i][j]))
    formula = f"Exp(-reduce(lambda a,b : Factorize_Impl(a,b[0],b[1]),{tmp}))"
    end = time.time()
    print("time for building kernel:", end-start)
    return formula




from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.code_gen_utils import c_variable, pointer, c_array, c_for_loop, c_zero_float
class SoftDTW(Operation):
    string_id = "SoftDTW"
    def __init__(self, x, y, gamma, params=()):
        # N.B. params keyword is used for compatibility with base class, but should always equal ()
        if params != ():
            KeOps_Error("There should be no parameter.")
        # x is vector of size n, y is vector of size m,
        # output is scalar
        if gamma.dim != 1:
            KeOps_Error("input gamma should be scalar")
        super().__init__(x, y, gamma)
        self.n = x.dim
        self.m = y.dim
        self.dim = 1

    def Op(self, out, table, x, y, gamma):
        dtype = x.dtype
        n,m = self.n, self.m
        code = f"""
            {dtype} ra[{n}], rb[{n}], *rj, *rjm1, dij;
            for (int j=0; j<{m}; j++)
            {{
                if (j%2) {{ rj=ra; rjm1=rb; }} else {{ rjm1=ra; rj=rb; }}
                for (int i=0; i<{n}; i++)
                {{
                    dij = ({x}[i] - {y}[j]);
                    dij *= dij;
                    if ((i == 0) && (j == 0))
                        rj[i] = dij;
                    else if (i==0)
                        rj[i] = dij + rjm1[i];
                    else if (j == 0)
                        rj[i] = dij + rj[i-1];
                    else
                        rj[i] = dij - (*{gamma}) * log( exp(-rjm1[i-1]/(*{gamma})) + exp(-rj[i-1]/(*{gamma})) + exp(-rjm1[i]/(*{gamma})) );
                    //printf("i,j,rj[i]=%d,%d,%f\\n",i,j,rj[i]);
                }}
            }}
            *{out} = rj[{n}-1];
            //printf("*out=%f\\n",*out);
                """
        return code
import builtins
builtins.SoftDTW = SoftDTW

def SoftDTW_keops_custom(x,y,gamma):
    start = time.time()
    end = time.time()
    print("time for building kernel:", end-start)
    return SoftDTW(x,y,gamma).__repr__()



def fun(x, y, gamma, backend):
    if backend == "keopscore":
        x = x.view((x.shape[0],x.shape[2]))
        y = y.view((y.shape[1],y.shape[2]))
        gamma = gamma.view((1,1))
        xi = Var(0,x.shape[1],0)
        yj = Var(1,y.shape[1],1)
        g = Var(2,1,2)
        aliases = [item.__repr__() for item in (xi,yj,g)]
        formula = SoftDTW_keopscore(xi,yj,g)
        Kxy = Genred(formula, aliases, reduction_op="Sum", axis=1)
        res = Kxy(x,y,gamma)
    elif backend == "keops_custom":
        x = x.view((x.shape[0],x.shape[2]))
        y = y.view((y.shape[1],y.shape[2]))
        gamma = gamma.view((1,1))
        xi = Var(0,x.shape[1],0)
        yj = Var(1,y.shape[1],1)
        g = Var(2,1,2)
        aliases = [item.__repr__() for item in (xi,yj,g)]
        formula = f"Exp(-{SoftDTW_keops_custom(xi,yj,g)})"
        Kxy = Genred(formula, aliases, reduction_op="Sum", axis=1)
        res = Kxy(x,y,gamma)
    elif backend == "lazytensor":
        Sxy = SoftDTW_keops(x,y,gamma)
        Kxy = (-Sxy).exp()
        res = Kxy.sum(dim=1)
    elif backend == "lazytensor_naive":
        Sxy = SoftDTW_keops_naive(x,y,gamma)
        Kxy = (-Sxy).exp()
        res = Kxy.sum(dim=1)
    elif backend == "torch":
        Sxy = SoftDTW_torch(x,y,gamma)
        Kxy = (-Sxy).exp()
        res = Kxy.sum(dim=1)
    return res


out = []
for backend in backends:
    print("**************************")
    print(f"Testing {backend} version")
    if do_warmup:
        start = time.time()
        fun(x[: min(M, 100), :, :], y[:, : min(N, 100), :], gamma, backend)
        end = time.time()
        print("time for compilation and first call:", end-start)
        fun(x[: min(M, 100), :, :], y[:, : min(N, 100), :], gamma, backend)
    start = time.time()
    out.append(fun(x, y, gamma, backend).squeeze())
    end = time.time()
    # print(out[-1].squeeze()[:10])
    print("time for " + backend + ":", end - start)

print("******")

if len(out) > 1:
    for k in range(1,len(out)):
        print(f"relative error {backends[k]} vs {backends[0]}:", (torch.norm(out[0] - out[k]) / torch.norm(out[0])).item())


if test_grad:
    print("Testing grads")
    out_g = []
    for k, backend in enumerate(backends):
        print("**************************")
        print(f"Testing {backend} version")
        start = time.time()
        out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [x])[0])
        end = time.time()
        print("time for " + backend + " (grad):", end - start)

    if len(out_g) > 1:
        for k in range(1,len(out)):
            print(f"relative error grad {backends[k]} vs {backends[0]}:", (torch.norm(out_g[0] - out_g[k]) / torch.norm(out_g[0])).item())

