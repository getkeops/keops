from keopscore.formulas.Chunkable_Op import Chunkable_Op
from keopscore.formulas.maths.Sum import Sum
from keopscore.utils.code_gen_utils import (
    c_zero_float,
    VectApply,
)
from keopscore.utils.math_functions import keops_fma
from keopscore.utils.misc_utils import KeOps_Error

##########################
#####    Scalprod     ####
##########################


class Scalprod_Impl(Chunkable_Op):
    string_id = "Scalprod"
    print_spec = "|", "mid", 3
    linearity_type = "one"

    dim = 1

    def __init__(self, fa, fb, params=()):
        # N.B. params keyword is used for compatibility with base class, but should always equal ()
        if params != ():
            KeOps_Error("There should be no parameter.")
        # Output dimension = 1, provided that FA::DIM = FB::DIM
        self.dimin = fa.dim
        if self.dimin != fb.dim:
            KeOps_Error("Dimensions must be the same for Scalprod")
        super().__init__(fa, fb)

    def Op(self, out, table, arga, argb):
        return out.assign(c_zero_float) + VectApply(self.ScalarOp, out, arga, argb)

    def ScalarOp(self, out, arga, argb):
        return out.assign(keops_fma(arga, argb, out))

    def DiffT(self, v, gradin):
        fa, fb = self.children
        return gradin * (fa.DiffT(v, fb) + fb.DiffT(v, fa))

    def initacc_chunk(self, acc):
        return f"*{acc.id} = 0.0f;\n"

    def acc_chunk(self, acc, out):
        return f"*{acc.id} += *{out.id};\n"


def Scalprod(arg0, arg1):
    from keopscore.formulas.maths.Mult import Mult_Impl

    if arg0.dim == 1:
        return arg0 * Sum(arg1)
    if arg1.dim == 1:
        return Sum(arg0) * arg1
    if arg0 == arg1:
        return Sum(arg0**2)
    if isinstance(arg0, Mult_Impl):
        u, v = arg0.children
        if u.dim == 1:
            return u * (v | arg1)
        if v.dim == 1:
            return v * (u | arg1)
    if isinstance(arg1, Mult_Impl):
        u, v = arg1.children
        if u.dim == 1:
            return u * (v | arg0)
        if v.dim == 1:
            return v * (u | arg0)
    return Scalprod_Impl(arg0, arg1)
