from keops.python_engine.utils.code_gen_utils import c_zero_float, cast_to
from keops.python_engine.formulas.reductions.Reduction import Reduction


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
        return f"{tmp.id} += {cast_to(tmp.dtype)}({xi.id});"

    def KahanScheme(self, acc, xi, tmp):
        return f"""
                        #pragma unroll
                        for (int k=0; k<{self.dim}; k++) 
                        {{
                            {acc.dtype} a = ({acc.dtype}) ({xi.id}[k] - {tmp.id}[k]);
                            {acc.dtype} b = {acc.id}[k] + a;
                            {tmp.id}[k] = ({tmp.dtype}) ((b - {acc.id}[k]) - a);
                            {acc.id}[k] = b;
                        }}
                """

    def DiffT(self, v, gradin, f0=None):
        from keops.python_engine.reductions import Grad
        return Sum_Reduction(Grad(self.formula, v, gradin), v.cat % 2)
