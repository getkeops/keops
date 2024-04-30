from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero_Impl


##########################
######    Square     #####
##########################


class Square_Impl(VectorizedScalarOp):
    """the square vectorized operation"""

    string_id = "Square"
    print_fun = lambda x: f"{x}**2"
    print_level = 1

    def ScalarOp(self, out, arg):
        return out.assign(arg) + out.mul_assign(out)

    @staticmethod
    def Derivative(f):
        return 2 * f


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Square(arg):
    if isinstance(arg, Zero_Impl):
        return arg
    else:
        return Square_Impl(arg)
