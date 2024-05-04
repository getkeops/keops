from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero, Zero_Impl
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.unique_object import unique_object

##########################
######    SumT       #####
##########################


class SumT_Impl(Operation):
    pass


class SumT_Impl_Factory(metaclass=unique_object):

    def __init__(self, dim):

        class Class(SumT_Impl):

            # the adjoint of the summation operation

            string_id = "SumT"
            linearity_type = "all"

            def __init__(self, arg):
                super().__init__(arg)
                self.dim = dim

            def Op(self, out, table, arg):
                return out.assign(arg.value)

            def DiffT_fun(self, v, gradin):
                from keopscore.formulas.maths.Sum import Sum

                f = self.children[0]
                return f.DiffT(v, Sum(gradin))

        self.Class = Class

    def __call__(self, f):
        return self.Class(f)


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def SumT(arg, dim):
    if arg.dim != 1:
        KeOps_Error("dimension of argument must be 1 for SumT operation")
    elif isinstance(arg, Zero_Impl):
        return Zero(dim)
    else:
        return SumT_Impl_Factory(dim)(arg)
