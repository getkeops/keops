from keopscore.utils.code_gen_utils import (
    c_zero_float,
    c_for_loop,
    c_variable,
    new_c_varname,
)
from keopscore.formulas.reductions.Reduction import Reduction


class Sum_Reduction(Reduction):
    """Sum reduction class"""

    string_id = "Sum_Reduction"

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim  # dimension of final output of reduction
        self.dimred = self.dim  # dimension of inner reduction variables
        self.dim_kahan = self.dim

    def InitializeReduction(self, tmp):
        # Returns C++ code to be used at initialization phase of the reduction.
        # Here it consists in setting the output array to zero.
        return tmp.assign(c_zero_float)

    def ReducePairScalar(self, tmp, xi):
        # Subroutine of ReducePairShort and ReducePair methods.
        # Returns C++ code that implements the "+=" accumulation operation of the sum reduction
        return tmp.add_assign(xi)

    def KahanScheme(self, acc, xi, tmp):
        loop, k = c_for_loop(0, self.dim, 1, pragma_unroll=True)
        a = c_variable(acc.dtype, new_c_varname("a"))
        b = c_variable(acc.dtype, new_c_varname("b"))
        return loop(
            a.declare_assign(xi[k] - tmp[k])
            + b.declare_assign(acc[k] + a)
            + tmp[k].assign((b - acc[k]) - a)
            + acc[k].assign(b)
        )

    def DiffT(self, v, gradin, f0=None):
        from keopscore.formulas.autodiff import Grad

        return Sum_Reduction(Grad(self.formula, v, gradin), v.cat % 2)
