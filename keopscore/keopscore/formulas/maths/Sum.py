from keopscore.formulas.Chunkable_Op import Chunkable_Op
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.code_gen_utils import c_zero_float, VectApply
from keopscore.formulas.maths.Square import Square_Impl


##########################
######    Sum        #####
##########################


class Sum_Impl(Chunkable_Op):
    # the summation operation

    string_id = "Sum"
    linearity_type = "all"

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
    from keopscore.formulas.maths.Mult import Mult_Impl

    if arg.dim == 1:
        return arg
    elif isinstance(arg, Zero):
        return Zero(1)
    elif isinstance(arg, Mult_Impl) and arg.children[0].dim == 1:
        # Sum(f*g) -> f*Sum(g) if f.dim=1
        return arg.children[0] * Sum(arg.children[1])
    elif isinstance(arg, Mult_Impl) and arg.children[1].dim == 1:
        # Sum(f*g) -> Sum(f)*g if g.dim=1
        return Sum(arg.children[0]) * arg.children[1]
    elif isinstance(arg, Mult_Impl):
        # Sum(f*g) -> f|g
        f, g = arg.children
        return f | g
    else:
        return Sum_Impl(arg)
