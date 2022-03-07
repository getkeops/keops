from keopscore.formulas.maths import Concat, Exp, Extract
from keopscore.formulas.reductions.Reduction import Reduction
from keopscore.formulas.reductions.Sum_Reduction import Sum_Reduction
from keopscore.formulas.variables.IntCst import IntCst
from keopscore.utils.code_gen_utils import (
    neg_infinity,
    c_zero_float,
    new_c_varname,
    c_variable,
    c_for_loop,
)
from keopscore.utils.math_functions import keops_exp
from keopscore.utils.misc_utils import KeOps_Error


class Max_SumShiftExpWeight_Reduction(Reduction):
    """Implements the coupled reduction operation m_i=max_j f_ij, s_i=sum_j exp(f_ij-m_i) g_ij
    where f and g are two formulas. f must be scalar-valued.
    This reduction is the base for numerically stable computation of log-sum-exp and softmax type reductions."""

    string_id = "Max_SumShiftExpWeight_Reduction"

    def __init__(self, formulaF, tagIJ, formulaG=IntCst(1)):
        if formulaF.dim != 1:
            KeOps_Error(
                "Max_SumShiftExpWeight_Reduction requires first formula of dimension 1."
            )
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
            KeOps_Error("Not implemented.")

        tmpexp = c_variable(acc.dtype, new_c_varname("tmpexp"))
        loop, k = c_for_loop(1, self.dimred, 1, pragma_unroll=True)
        return f"""       
                      {tmpexp.declare()}
                      if ({acc.id}[0] > {xi.id}[0]) {{ // =  exp(m)  * (s + s'*exp(m'-m))   if m > m'
                        {tmpexp.assign(keops_exp(xi[0]-acc[0]))}
                        {loop(acc[k].add_assign(xi[k]*tmpexp))}
                      }} else {{             // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
                        {tmpexp.assign(keops_exp(acc[0]-xi[0]))}
                        {loop(acc[k].assign(xi[k]+tmpexp*acc[k]))}
                        {acc[0].assign(xi[0])}
                      }}
              """

    def ReducePairShort(self, acc, xi, ind):
        return self.ReducePair(acc, xi)

    def KahanScheme(self, acc, xi, tmp):
        if xi.dtype == "half2":
            KeOps_Error("Not implemented.")
        tmpexp = c_variable(acc.dtype, new_c_varname("tmpexp"))
        loop, k = c_for_loop(1, self.dimred, 1, pragma_unroll=True)
        a = c_variable(acc.dtype, new_c_varname("a"))
        b = c_variable(acc.dtype, new_c_varname("b"))
        u = c_variable(acc.dtype, new_c_varname("u"))
        return f"""
                        {tmpexp.declare()}
                        if ({acc.id}[0] > {xi.id}[0])    // =  exp(m)  * (s + s'*exp(m'-m))   if m > m'
                        {{      
                            {tmpexp.assign(keops_exp(xi[0]-acc[0]))}
                            {loop( a.declare_assign(xi[k]*tmpexp-tmp[k-1]) + b.declare_assign(acc[k]+a) + tmp[k-1].assign((b-acc[k])-a) + acc[k].assign(b))}
                        }} 
                        else      // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
                        {{             
                            {tmpexp.assign(keops_exp(acc[0]-xi[0]))}
                            {loop( u.declare_assign(tmpexp*acc[k]) + a.declare_assign(xi[k]-tmpexp*tmp[k-1]) + b.declare_assign(u+a) + tmp[k-1].assign((b-u)-a) + acc[k].assign(b))}
                            {acc[0].assign(xi[0])}
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
        from keopscore.formulas.autodiff import Grad

        M = Extract(MS, 0, self.formulaF.dim)
        S = Extract(gradin, self.formulaF.dim, self.formulaG.dim)
        return Grad(
            Sum_Reduction(Exp(self.formulaF - M) * self.formulaG, self.tagI), v, S
        )


Max_SumShiftExp_Reduction = Max_SumShiftExpWeight_Reduction
