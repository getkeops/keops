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
    minargs = reduce(lambda x,y:torch.min(x,y), args)
    if gamma>0:
        minargs -= gamma * sum(((minargs-arg)/gamma).exp() for arg in args).log() 
    return minargs

def SoftDTW_torch(x, y, gamma):
    n, m = x.shape[1], y.shape[1]
    x, y = x[:,None,:], y[None,:,:]
    rjm1 = [torch.tensor(torch.inf, device=device_id) for _ in range(n+1)]
    rjm1[0] = torch.tensor(0., device=device_id)
    torchinf = torch.tensor(torch.inf, device=device_id)
    for j in range(1,m+1):
        rim1j = torchinf
        for i in range(1,n+1):
            rij = (x[:,:,i-1]-y[:,:,j-1])**2 + softmin((rjm1[i], rjm1[i-1], rim1j), gamma)
            rjm1[i-1] = rim1j
            rim1j = rij
        rjm1[i] = rij
    return rij

####################################################################
# SoftDTW operation in keops for squared difference dissimilarity
####################################################################

from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.code_gen_utils import c_variable, pointer, c_array, c_for_loop, c_zero_float
from keopscore.utils.code_gen_utils import use_pragma_unroll

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
        n,m = self.n, self.m
        code = f"""
            #define MIN2(a,b) fminf(a,b) //(((a)<(b))?(a):(b))
            #define MIN3(a,b,c) MIN2(MIN2(a,b),c)
            
            {dtype} rjm1[{n}], rim1j, rij, min;
            // j=0, i=0
            rij = {x}[0] - {y}[0];
            rij *= rij;
            rim1j = rij;

            // j=0, i=1...n-1
            {use_pragma_unroll()}
            for (int i=1; i<{n}; i++)
            {{
                rij = {x}[i] - {y}[0];
                rij *= rij;
                rij += rim1j;
                rjm1[i-1] = rim1j;
                rim1j = rij;
            }}
            rjm1[{n}-1] = rij;

            {use_pragma_unroll()}
            for (int j=1; j<{m}; j++)
            {{
                // j=1...m-1, i=0
                rij = {x}[0] - {y}[j];
                rij *= rij;
                rij += rjm1[0];
                rim1j = rij;

                {use_pragma_unroll()}
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
        x, y, gamma = self.children
        n,m = self.n, self.m
        if v in gamma.Vars_:
            KeOps_Error("autograd wrt gamma in SoftDTW operation not implemented.")
        grad = GradSoftDTW(x, y, gamma)
        gradx = Extract(grad,0,n)
        grady = Extract(grad,n,n+m)
        return x.DiffT(v, gradx) + y.DiffT(v, grady)

import builtins
builtins.SoftDTW_L2 = SoftDTW_L2



####################################################################
# SoftDTW operation in keops (generic, from dissimilarity matrix)
####################################################################

from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.code_gen_utils import c_variable, pointer, c_array, c_for_loop, c_zero_float
from keopscore.utils.code_gen_utils import use_pragma_unroll



class GradSoftDTW(Operation):
    string_id = "GradSoftDTW"
    def __init__(self, x, y, gamma, params=()):
        # x is vector of size n, y is vector of size m, gamma is scalar,
        # output is of size n+m, corresponding to concatenation of grads wrt x and y
        if gamma.dim != 1:
            KeOps_Error("input gamma should be scalar")
        n,m = x.dim, y.dim
        super().__init__(x, y, gamma, params=())
        self.n = n
        self.m = m
        self.dim = n+m

    def Op(self, out, table, x, y, gamma):
        dtype = x.dtype
        n,m = self.n, self.m
        code = f"""
            #define MIN2(a,b) fminf(a,b) //(((a)<(b))?(a):(b))
            #define MIN3(a,b,c) MIN2(MIN2(a,b),c)
            
            {dtype} r[{n*m}], min, d;

            // Forward pass to fill in r matrix

            // j=0, i=0
            d = {x}[0] - {y}[0];
            r[0] = d*d;

            // j=0, i=1...n-1
            {use_pragma_unroll()}
            for (int i=1; i<{n}; i++)
            {{
                d = {x}[i] - {y}[j];
                r[i] = d*d + r[i-1];
            }}                

            {use_pragma_unroll()}
            for (int j=1; j<{m}; j++)
            {{
                // j=1...m-1, i=0
                d = {x}[0] - {y}[j];
                r[j*{n}] = d*d + r[(j-1)*{n}];

                {use_pragma_unroll()}
                for (int i=1; i<{n}; i++)
                {{
                    // j=1...m-1, i=1...n-1
                    d = {x}[i] - {y}[j];
                    r[j*{n}+i] = d*d;
                    min = MIN3(r[(j-1)*{n}+i-1],r[(j-1)*{n}+i],r[j*{n}+i-1]);
                    r[j*{n}+i] += min - {gamma}[0] * log( exp((min-r[(j-1)*{n}+i-1])/{gamma}[0]) + exp((min-r[j*{n}+i-1])/{gamma}[0]) + exp((min-r[(j-1)*{n}+i])/{gamma}[0]) );
                }}
            }}

            printf("end of forward = %f\\n", r[j*{n}+i]);

            // backward pass

            {dtype} ejp1[{n}], eip1j, eij, a, b, c;

            // j=m-1, i=n-1
            eij = 1.0;
            eip1j = eij;






            
            
            // j=m-1, i=n-2..0
            {use_pragma_unroll()}
            for (int i={n-2}; i>=0; i--)
            {{
                d = {x}[i+1] - {y}[{m-1}];
                a = exp((r[{(m-1)*n}+i+1]-r[{(m-1)*n}+i]-d*d)/{gamma}[0]);
                eij = a * eip1j;
                ejp1[i+1] = eip1j;
                eip1j = eij;
            }}
            ejp1[0] = eij;

            {use_pragma_unroll()}
            for (int j={m-2}; j>=0; j--)
            {{
                // j=m-2..0, i=n-1
                d = {x}[{n-1}] - {y}[j+1];
                b = exp((r[(j+1)*{n}+{n-1}]-r[j*{n}+{n-1}]-d*d)/{gamma}[0]);
                eip1j = b * ejp1[{n-1}];

                {use_pragma_unroll()}
                for (int i={n-2}; i>=0; i--)
                {{
                    // j=m-2..0, i=n-2..0
                    d = {x}[i+1] - {y}[j];
                    a = exp((r[j*{n}+i+1]-r[j*{n}+i]-d*d)/{gamma}[0]);
                    d = {x}[i] - {y}[j+1];
                    b = exp((r[(j+1)*{n}+i]-r[j*{n}+i]-d*d)/{gamma}[0]);
                    d = {x}[i+1] - {y}[j+1];
                    c = exp((r[(j+1)*{n}+i+1]-r[j*{n}+i]-d*d)/{gamma}[0]);
                    eij = a * eip1j + b * ejp1[i] + c * ejp1[i+1];
                    ejp1[i+1] = eip1j;
                    eip1j = eij;
                }}
                ejp1[0] = eij;
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
    Sxy = SoftDTW_torch(x,y,gamma)
    Kxy = (-Sxy).exp()
    return Kxy.sum(dim=1)

def fun_keops(x, y, gamma):
    n,m = x.shape[1], y.shape[1]
    formula = "Exp(-SoftDTW_L2(x,y,gamma))"
    aliases = [f"x=Vi({n})", f"y=Vj({m})", "gamma=Pm(1)"]
    Kxy = Genred(formula, aliases, reduction_op="Sum", axis=1)
    return Kxy(x,y,gamma.view((1,1)))

def fun_lazytensor(x, y, gamma):
    x = LazyTensor(x[:,None,:])
    y = LazyTensor(y[None,:,:])
    sdtw = x.softdtw_l2(y,gamma)
    K = (-sdtw).exp()
    return K.sum(axis=1)

def fun_lazytensor_diffmatrix(x, y, gamma):
    x = LazyTensor(x[:,None,:])
    y = LazyTensor(y[None,:,:])
    dist_l2 = x.difference_matrix(y)**2
    sdtw = dist_l2.softdtw(gamma, input_shape=(x.ndim, y.ndim))
    K = (-sdtw).exp()
    return K.sum(axis=1)

##################################
# test
##################################

#funs = (fun_torch, fun_keops, fun_lazytensor, fun_lazytensor_diffmatrix)
funs = (fun_torch, fun_lazytensor_diffmatrix)
out = []
for fun in funs:
    print("**************************")
    print("Testing " + fun.__name__)
    if do_warmup:
        fun(x[:100,:], y[:100,:], gamma)
        fun(x[:100,:], y[:100,:], gamma)
    start = time.time()
    out.append(fun(x, y, gamma).squeeze())
    end = time.time()
    print("time for " + fun.__name__ + ":", end - start)

print("******")

if len(out) > 1:
    for k in range(1,len(out)):
        print(f"relative error {funs[k].__name__} vs {funs[0].__name__}:", (torch.norm(out[0] - out[k]) / torch.norm(out[0])).item())


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
        for k in range(1,len(out)):
            print(f"relative error grad {funs[k].__name__} vs {funs[0].__name__}:", (torch.norm(out_g[0] - out_g[k]) / torch.norm(out_g[0])).item())

