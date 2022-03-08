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

    dim = 1

    def __init__(self, fa, fb):
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
    if arg0.dim == 1:
        return arg0 * Sum(arg1)
    elif arg1.dim == 1:
        return Sum(arg0) * arg1
    else:
        return Scalprod_Impl(arg0, arg1)
