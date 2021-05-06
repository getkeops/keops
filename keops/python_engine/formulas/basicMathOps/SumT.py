from keops.python_engine.code_gen_utils import c_variable
from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.Zero import Zero


##########################
######    SumT       #####
##########################


class SumT_(Operation):
    # the adjoint of the summation operation
    string_id = "SumT"

    def __init__(self, arg, dim):
        super().__init__(arg)
        self.dim = dim
        self.params = (dim,)

    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        value_arg = c_variable(arg.dtype, f"*{arg.id}")
        return out.assign(value_arg)

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.basicMathOps.Sum import Sum
        f = self.children[0]
        return f.Grad(v, Sum(gradin))


def SumT(arg, dim):
    if arg.dim != 1:
        raise ValueError("dimension of argument must be 1 for SumT operation")
    elif isinstance(arg, Zero):
        return Zero(dim)
    else:
        return SumT_(arg, dim)