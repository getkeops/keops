from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.variables.Zero import Zero
from keops.python_engine.utils.code_gen_utils import value


##########################
######    SumT       #####
##########################


class SumT_Impl(Operation):
    # the adjoint of the summation operation

    string_id = "SumT"

    def __init__(self, arg, dim):
        super().__init__(arg, params=(dim,))
        self.dim = dim

    def Op(self, out, table, arg):
        return out.assign(value(arg))

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.maths.Sum import Sum

        f = self.children[0]
        return f.DiffT(v, Sum(gradin))


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def SumT(arg, dim):
    if arg.dim != 1:
        raise ValueError("dimension of argument must be 1 for SumT operation")
    elif isinstance(arg, Zero):
        return Zero(dim)
    else:
        return SumT_Impl(arg, dim)
