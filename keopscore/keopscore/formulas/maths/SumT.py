from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.meta_toolbox import c_fixed_size_array

##########################
######    SumT       #####
##########################


class SumT_Impl(Operation):
    # the adjoint of the summation operation

    string_id = "SumT"
    linearity_type = "all"

    def __init__(self, arg, dim=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if dim is None:
            # here params should be a tuple containing one single integer
            (dim,) = params
        super().__init__(arg, params=(dim,))
        self.dim = dim

    def Op(self, out, table, arg):
        return out.assign(arg.value)

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.Sum import Sum

        f = self.children[0]
        return f.DiffT(v, Sum(gradin))


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def SumT(arg, dim):
    if arg.dim != 1:
        KeOps_Error("dimension of argument must be 1 for SumT operation")
    elif isinstance(arg, Zero):
        return Zero(dim)
    else:
        return SumT_Impl(arg, dim)
