from keops.python_engine.utils.code_gen_utils import neg_infinity
from keops.python_engine.formulas.reductions.Reduction import Reduction


class Max_Reduction(Reduction):
    """Implements the max reduction operation : for each i or each j, find the
    maximal value of Fij operation is vectorized: if Fij is vector-valued, max is computed for each dimension."""

    string_id = "Max_Reduction"

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim  # dimension of final output of reduction
        self.dimred = self.dim  # dimension of inner reduction variables

    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        # Here it consists in setting the output array to -infinity.
        return acc.assign(neg_infinity(acc.dtype))

    def ReducePairScalar(self, acc, xi):
        # Subroutine of ReducePairShort and ReducePair methods.
        if xi.dtype == "half2":
            raise ValueError("not implemented")
        return f"""
                    if({xi.id}>{acc.id}) 
                        {acc.id} = {xi.id};
                """
