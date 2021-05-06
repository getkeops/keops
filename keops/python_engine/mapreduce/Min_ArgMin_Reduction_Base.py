from keops.python_engine.code_gen_utils import infinity, c_zero_float, VectApply
from keops.python_engine.mapreduce.Reduction import Reduction


class Min_ArgMin_Reduction_Base(Reduction):
    """min+argmin reduction : base class"""

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)

        # We work with a (values,indices) vector
        self.dimred = 2 * formula.dim  # dimension of inner reduction variables

    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        acc_min, acc_argmin = acc.split(self.dim, self.dim)
        return acc_min.assign(infinity(acc.dtype)) + acc_argmin.assign(c_zero_float)

    def ReducePairScalar(self, acc_val, acc_ind, xi, ind):
        # Subroutine of ReducePairShort and ReducePair methods.
        if xi.dtype == "half2":
            raise ValueError("not implemented")
        return f"""
                    if({xi.id}<{acc_val.id}) 
                    {{
                        {acc_val.id} = {xi.id};
                        {acc_ind.id} = {ind.id};
                    }}
                """

    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        acc_val, acc_ind = acc.split(self.dim, self.dim)
        xi_val, xi_ind = xi.split(self.dim, self.dim)
        return VectApply(self.ReducePairScalar, acc_val, xi_val, xi_ind)

    def ReducePairShort(self, acc, xi, ind):
        if xi.dtype == "half2":
            raise ValueError("not implemented")
            half2_val = c_variable("half2_ind")
            string = half2_val.declare_assign(f"__floats2half2_rn(2*{ind()},2*{ind()}+1)")
        acc_val, acc_ind = acc.split(self.dim, self.dim)
        ind_arr = c_array(xi.dtype, 1, f"(&{ind.id})")
        return VectApply(self.ReducePairScalar, acc_val, acc_ind, xi, ind_arr)
