from keopscore.formulas.Chunkable_Op import Chunkable_Op
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.code_gen_utils import c_zero_float, VectApply


##########################
######    Sum        #####
##########################


class Sum_Impl(Chunkable_Op):
    # the summation operation

    string_id = "Sum"

    dim = 1

    def Op(self, out, table, arg):
        return out.assign(c_zero_float) + VectApply(self.ScalarOp, out, arg)

    def ScalarOp(self, out, arg):
        return out.add_assign(arg)

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.SumT import SumT

        f = self.children[0]
        return f.DiffT(v, SumT(gradin, f.dim))

    def initacc_chunk(self, acc):
        return f"*{acc.id} = 0.0f;\n"

    def acc_chunk(self, acc, out):
        return f"*{acc.id} += *{out.id};\n"


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Sum(arg):
    if isinstance(arg, Zero):
        return Zero(1)
    else:
        return Sum_Impl(arg)
