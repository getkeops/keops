from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.maths.Mult import Mult_Impl
from keopscore.formulas.variables.IntCst import IntCst_Impl, IntCst
from keopscore.formulas.variables.RatCst import RatCst_Impl, RatCst


##########################
######    Minus      #####
##########################


class Minus_Impl(VectorizedScalarOp):
    """the "minus" vectorized operation"""

    string_id = "Minus"
    print_spec = "-", "pre", 2
    linearity_type = "all"

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = -{arg.id};\n"

    def DiffT(self, v, gradin):
        f = self.children[0]
        return -f.DiffT(v, gradin)

    # parameters for testing the operation (optional)
    nargs = 1  # number of arguments
    torch_op = "torch.neg"  # equivalent PyTorch operation


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Minus(arg):
    if isinstance(arg, Zero):
        return arg
    elif isinstance(arg, Minus_Impl):
        # -(-f) -> f
        return arg.children[0]
    elif isinstance(arg, IntCst_Impl):
        # -(n) -> (-n)
        return IntCst(-arg.params[0])
    elif isinstance(arg, RatCst_Impl):
        # -(p/q) -> (-p)/q
        p, q = arg.params
        return RatCst(-p, q)
    elif isinstance(arg, Mult_Impl) and isinstance(
        arg.children[0], (IntCst_Impl, RatCst_Impl)
    ):
        r, g = arg.children
        # -(r*g) -> (-r)*g
        return (-r) * g
    else:
        return Minus_Impl(arg)
