from keopscore.utils.code_gen_utils import (
    infinity,
    c_zero_float,
    VectApply,
    c_if,
)
from keopscore.formulas.reductions.Reduction import Reduction
from keopscore.utils.misc_utils import KeOps_Error


class Min_ArgMin_Reduction_Base(Reduction):
    """min+argmin reduction : base class"""

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)

        # We work with a (values,indices) vector
        self.dimred = 2 * formula.dim  # dimension of inner reduction variables

    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        dim = self.formula.dim
        acc_min, acc_argmin = acc.split(dim, dim)
        return acc_min.assign(infinity(acc.dtype)) + acc_argmin.assign(c_zero_float)

    def ReducePairScalar(self, acc_val, acc_ind, xi, ind):
        # Subroutine of ReducePairShort and ReducePair methods.
        if xi.dtype == "half2":
            KeOps_Error("not implemented")
        return c_if(xi < acc_val, acc_val.assign(xi) + acc_ind.assign(ind))

    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        dim = self.formula.dim
        acc_val, acc_ind = acc.split(dim, dim)
        xi_val, xi_ind = xi.split(dim, dim)
        return VectApply(self.ReducePairScalar, acc_val, xi_val, xi_ind)

    def ReducePairShort(self, acc, xi, ind):
        if xi.dtype == "half2":
            KeOps_Error("not implemented")
            half2_val = c_variable("half2_ind")
            string = half2_val.declare_assign(
                f"__floats2half2_rn(2*{ind()},2*{ind()}+1)"
            )
        dim = self.formula.dim
        acc_val, acc_ind = acc.split(dim, dim)
        return VectApply(self.ReducePairScalar, acc_val, acc_ind, xi, ind)
