from keops.python_engine.formulas.maths.Concat import Concat
from keops.python_engine.formulas.maths.Exp import Exp
from keops.python_engine.formulas.maths.Extract import Extract
from keops.python_engine.formulas.reductions.Reduction import Reduction
from keops.python_engine.formulas.reductions.Sum_Reduction import Sum_Reduction
from keops.python_engine.formulas.variables.IntCst import IntCst
from keops.python_engine.utils.code_gen_utils import neg_infinity, c_zero_float


class Max_SumShiftExpWeight_Reduction(Reduction):
    """Implements the coupled reduction operation m_i=max_j f_ij, s_i=sum_j exp(f_ij-m_i) g_ij
    where f and g are two formulas. f must be scalar-valued.
    This reduciton is the base for numerically stable computation of log-sum-exp and softmax type reductions."""

    string_id = "Max_SumShiftExpWeight_Reduction"

    def __init__(self, formulaF, tagIJ, formulaG=IntCst(1)):
        if formulaF.dim != 1:
            raise ValueError("Max_SumShiftExpWeight_Reduction requires first formula of dimension 1.")
        super().__init__(Concat(formulaF, formulaG), tagIJ)
        self.formulaF = formulaF
        self.formulaG = formulaG
        self.dim = formulaF.dim + formulaG.dim  # dimension of final output of reduction
        self.dimred = self.dim  # dimension of inner reduction variables

    def InitializeReduction(self, acc):
        """Returns C++ code to be used at initialization phase of the reduction.
        We fill empty cells with the neutral element of the reduction operation,
         (-inf,0) = e^{-inf} * 0 = 0"""
        m, s = acc.split(1, self.formulaG.dim)
        return m.assign(neg_infinity(acc.dtype)) + s.assign(c_zero_float)

    def ReducePair(self, acc, xi):
        """Returns C++ code that implements the update phase of the reduction.
        (m,s) + (m',s'), i.e. exp(m)*s + exp(m')*s'"""

        if xi.dtype == "half2":
            raise ValueError("Not implemented.")

        return f"""       
                      {acc.dtype} tmpexp;
                      if ({acc.id}[0] > {xi.id}[0]) {{ // =  exp(m)  * (s + s'*exp(m'-m))   if m > m'
                        tmpexp = exp({xi.id}[0] - {acc.id}[0]);
                        #pragma unroll
                        for (int k = 1; k < {self.dimred}; k++)
                          {acc.id}[k] += {xi.id}[k] * tmpexp;
                      }} else {{             // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
                        tmpexp = exp({acc.id}[0] - {xi.id}[0]);
                        #pragma unroll
                        for (int k = 1; k < {self.dimred}; k++)
                          {acc.id}[k] = {xi.id}[k] + tmpexp * {acc.id}[k];
                        {acc.id}[0] = {xi.id}[0];
                      }}
              """

    def ReducePairShort(self, acc, xi, ind):
        return self.ReducePair(acc, xi)

    def KahanScheme(self, acc, xi, tmp):
        if xi.dtype == "half2":
            raise ValueError("Not implemented.")
        return f"""
                        {acc.id}.dtype tmpexp;
                        if ({acc.id}[0] > {xi.id}[0])    // =  exp(m)  * (s + s'*exp(m'-m))   if m > m'
                        {{      
                            tmpexp = exp({xi.id}[0] - {acc.id}[0]);
                            #pragma unroll
                            for (int k=1; k<{self.dimred}; k++)
                            {{
                                {acc.dtype} a = {xi.id}[k] * tmpexp - {tmp.id}[k-1];
                                {acc.dtype} b = {acc.id}[k] + a;
                                {tmp.id}[k-1] = (b - {acc.id}[k]) - a;
                                {acc.id}[k] = b;
                            }}
                        }} 
                        else      // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
                        {{             
                            tmpexp = exp({acc.id}[0] - {xi.id}[0]);
                            #pragma unroll
                            for (int k = 1; k < {self.dimred}; k++)
                            {{
                                {acc.dtype} u = tmpexp * {acc.id}[k];
                                {acc.dtype} a = {xi.id}[k] - tmpexp * {tmp.id}[k-1];
                                {acc.dtype} b = u + a;
                                {tmp.id}[k-1] = (b - u) - a;
                                {acc.id}[k] = b;
                            }}
                            {acc.id}[0] = {xi.id}[0];
                        }}
                """

    def DiffT(self, v, gradin, MS):
        """
          // Beware: the formula that we use for the gradient is *only* valid
          // if the output [M,S] = Max_SumShiftExp(F,G) has been flattened through a
          // L = M + log(S) (Log-Sum-Exp) or a weighted Soft-Max
          // operation (as done by the Python bindings), and if
          // GRADIN = [Grad(L), Grad(L)/S ]
          // has been backpropagated from L.
        """
        from keops.python_engine.reductions import Grad
        M = Extract(MS, 0, self.formulaF.dim)
        S = Extract(gradin, self.formulaF.dim, self.formulaG.dim)
        return Grad(Sum_Reduction(Exp(self.formulaF - M) * self.formulaG, self.tagI), v, S)


Max_SumShiftExp_Reduction = Max_SumShiftExpWeight_Reduction
