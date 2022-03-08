from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp

from keopscore.formulas.variables.Zero import Zero


##########################
######    Square     #####
##########################


class Square_Impl(VectorizedScalarOp):
    """the square vectorized operation"""

    string_id = "Square"
    print_spec = "**2", "post", 1

    def ScalarOp(self, out, arg):
        return out.assign(arg * arg)

    @staticmethod
    def Derivative(f):
        return 2 * f


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Square(arg):
    if isinstance(arg, Zero):
        return arg
    else:
        return Square_Impl(arg)
