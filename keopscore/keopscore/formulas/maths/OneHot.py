from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import c_zero_float
from keopscore.utils.misc_utils import KeOps_Error

# //////////////////////////////////////////////////////////////
# ////       ONE-HOT REPRESENTATION : OneHot<F,DIM>         ////
# //////////////////////////////////////////////////////////////


class OneHot(Operation):

    string_id = "OneHot"

    def __init__(self, f, dim):
        if f.dim != 1:
            KeOps_Error("One-hot representation is only supported for scalar formulas.")
        if dim < 1:
            KeOps_Error("A one-hot vector should have length >= 1.")
        super().__init__(f)
        self.dim = dim
        self.params = (dim,)

    def Op(self, out, table, arg0):
        if out.dtype == "half2" and arg0.dtype == "half2":
            return f"""
                        #pragma unroll
                        for (int k = 0; k < {self.dim}; k++)
                            {out.id}[k] = __heq2(h2rint(*{arg0.id}),__float2half2_rn(k));
                    """
        else:
            string = out.assign(c_zero_float)
            string += f"""
                           {out.id}[(int)(*{arg0.id}+.5)] = 1.0;
                       """
            return string

    def DiffT(self, v, gradin):
        from keopscore.formulas.variables.Zero import Zero

        return Zero(v.dim)
