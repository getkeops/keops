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
from keopscore.utils.code_gen_utils import use_pragma_unroll
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.maths.Extract import Extract


class SoftDTW_SqDist(Operation):
    string_id = "SoftDTW_SqDist"

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
        n, m = self.n, self.m
        if v in gamma.Vars_:
            KeOps_Error(
                "autograd wrt gamma in SoftDTW_SqDist operation not implemented."
            )
        grad = GradSoftDTW_SqDist(x, y, gamma) * gradin
        gradx = Extract(grad, 0, n)
        grady = Extract(grad, n, m)
        return x.DiffT(v, gradx) + y.DiffT(v, grady)


class GradSoftDTW_SqDist(Operation):
    string_id = "GradSoftDTW_SqDist"

    def __init__(self, x, y, gamma, params=()):
        # x is vector of size n, y is vector of size m, gamma is scalar,
        # output is of size n+m, corresponding to concatenation of grads wrt x and y
        if gamma.dim != 1:
            KeOps_Error("input gamma should be scalar")
        n, m = x.dim, y.dim
        super().__init__(x, y, gamma, params=())
        self.n = n
        self.m = m
        self.dim = n + m

    def Op(self, out, table, x, y, gamma):
        dtype = x.dtype
        n, m = self.n, self.m
        code = f"""
            #define MIN2(a,b) fminf(a,b) //(((a)<(b))?(a):(b))
            #define MIN3(a,b,c) MIN2(MIN2(a,b),c)
            
            {dtype} r[{n*m}], min, d;
            {dtype}* gradx = {out};
            {dtype}* grady = {out}+{n};

            // Forward pass to fill in r matrix

            // j=0, i=0
            d = {x}[0] - {y}[0];
            r[0] = d*d;

            // j=0, i=1...n-1
            {use_pragma_unroll()}
            for (int i=1; i<{n}; i++)
            {{
                d = {x}[i] - {y}[0];
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

            // backward pass

            {dtype} ejp1[{n}], eip1j, eij, a, b, c;

            // j=m-1, i=n-1
            eij = 1.0;
            eip1j = eij;
            d = {x}[{n-1}] - {y}[{m-1}];
            gradx[{n-1}] = 2*d*eij;
            grady[{m-1}] = -2*d*eij;
    
            // j=m-1, i=n-2..0
            {use_pragma_unroll()}
            for (int i={n-2}; i>=0; i--)
            {{
                a = exp((r[{(m-1)*n}+i+1]-r[{(m-1)*n}+i]-d*d)/{gamma}[0]);
                eij = a * eip1j;
                ejp1[i+1] = eip1j;
                eip1j = eij;
                d = {x}[i] - {y}[{m-1}];
                gradx[i] = 2*d*eij;
                grady[{m-1}] -= 2*d*eij;
            }}
            ejp1[0] = eij;

            {use_pragma_unroll()}
            for (int j={m-2}; j>=0; j--)
            {{
                // j=m-2..0, i=n-1
                d = {x}[{n-1}] - {y}[j+1];
                b = exp((r[(j+1)*{n}+{n-1}]-r[j*{n}+{n-1}]-d*d)/{gamma}[0]);
                eij = b * ejp1[{n-1}];
                eip1j = eij;
                d = {x}[{n-1}] - {y}[j];
                gradx[{n-1}] += 2*d*eij;
                grady[j] = -2*d*eij;

                {use_pragma_unroll()}
                for (int i={n-2}; i>=0; i--)
                {{
                    // j=m-2..0, i=n-2..0
                    a = exp((r[j*{n}+i+1]-r[j*{n}+i]-d*d)/{gamma}[0]);
                    d = {x}[i] - {y}[j+1];
                    b = exp((r[(j+1)*{n}+i]-r[j*{n}+i]-d*d)/{gamma}[0]);
                    d = {x}[i+1] - {y}[j+1];
                    c = exp((r[(j+1)*{n}+i+1]-r[j*{n}+i]-d*d)/{gamma}[0]);
                    eij = a * eip1j + b * ejp1[i] + c * ejp1[i+1];
                    ejp1[i+1] = eip1j;
                    eip1j = eij;
                    d = {x}[i] - {y}[j];
                    gradx[i] += 2*d*eij;
                    grady[j] -= 2*d*eij;
                }}
                ejp1[0] = eij;
            }}
                """

        return code

    def DiffT(self, v, gradin):
        KeOps_Error("autograd for GradSoftDTW_SqDist operation not implemented.")
        pass
