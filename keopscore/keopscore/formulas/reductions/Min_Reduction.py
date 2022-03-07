from keopscore.utils.code_gen_utils import infinity, c_if
from keopscore.formulas.reductions.Reduction import Reduction
from keopscore.utils.misc_utils import KeOps_Error


class Min_Reduction(Reduction):
    """Implements the min reduction operation : for each i or each j, find the minimal value of Fij
    operation is vectorized: if Fij is vector-valued, min is computed for each dimension."""

    string_id = "Min_Reduction"

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim  # dimension of final output of reduction
        self.dimred = self.dim  # dimension of inner reduction variables

    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        # Here it consists in setting the output array to -infinity.
        return acc.assign(infinity(acc.dtype))

    def ReducePairScalar(self, acc, xi):
        # Subroutine of ReducePairShort and ReducePair methods.
        if xi.dtype == "half2":
            KeOps_Error("not implemented")
        return c_if(xi < acc, acc.assign(xi))
