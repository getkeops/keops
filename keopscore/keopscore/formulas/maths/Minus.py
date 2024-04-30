from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero_Impl
from keopscore.formulas.maths.Mult import Mult_Impl
from keopscore.formulas.variables.IntCst import IntCst_Impl, IntCst
from keopscore.formulas.variables.RatCst import RatCst_Impl, RatCst
from keopscore.utils.meta_toolbox.c_instruction import c_instruction_from_string
from keopscore.utils.math_functions import keops_minus

##########################
######    Minus      #####
##########################


class Minus_Impl(VectorizedScalarOp):
    """the "minus" vectorized operation"""

    string_id = "Minus"
    print_fun = lambda x: f"-{x}"
    print_level = 2
    linearity_type = "all"

    ScalarOpFun = keops_minus

    def DiffT(self, v, gradin):
        f = self.children[0]
        return -f.DiffT(v, gradin)

    # parameters for testing the operation (optional)
    nargs = 1  # number of arguments
    torch_op = "torch.neg"  # equivalent PyTorch operation


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Minus(arg):
    if isinstance(arg, Zero_Impl):
        return arg
    elif isinstance(arg, Minus_Impl):
        # -(-f) -> f
        return arg.children[0]
    elif isinstance(arg, IntCst_Impl):
        # -(n) -> (-n)
        return IntCst(-arg.val)
    elif isinstance(arg, RatCst_Impl):
        # -(p/q) -> (-p)/q
        p, q = arg.p, arg.q
        return RatCst(-p, q)
    elif isinstance(arg, Mult_Impl) and isinstance(
        arg.children[0], (IntCst_Impl, RatCst_Impl)
    ):
        r, g = arg.children
        # -(r*g) -> (-r)*g
        return (-r) * g
    else:
        return Minus_Impl(arg)
