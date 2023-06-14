from keopscore.formulas.Operation import Broadcast
from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Scalprod import Scalprod
from keopscore.formulas.maths.Square import Square
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.math_functions import keops_mul
from keopscore.formulas.variables.IntCst import IntCst_Impl
from keopscore.formulas.maths.SumT import SumT, SumT_Impl
from keopscore.utils.misc_utils import KeOps_Error

##########################
######    Mult       #####
##########################


class Mult_Impl(VectorizedScalarOp):
    """the binary multiply operation"""

    string_id = "Mult"
    print_spec = "*", "mid", 3

    ScalarOpFun = keops_mul

    @staticmethod
    def Derivative(a, b):
        return b, a

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments
    torch_op = "torch.mul"  # equivalent PyTorch operation


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Mult(arg0, arg1, shapes=None):
    if isinstance(arg0, Zero):
        return Broadcast(arg0, arg1.dim)
    elif isinstance(arg1, Zero):
        return Broadcast(arg1, arg0.dim)
    elif isinstance(arg0, IntCst_Impl):
        if arg0.val == 1:
            # 1*f -> f
            return arg1
        if arg0.val == -1:
            # -1*f -> -f
            return -arg1
        elif isinstance(arg1, IntCst_Impl):
            # m*n -> mn
            return IntCst_Impl(arg0.val * arg1.val)
    if isinstance(arg1, IntCst_Impl):
        # f*n -> n*f (bringing integers to the left)
        return Mult(arg1, arg0)
    elif isinstance(arg1, Mult_Impl) and isinstance(arg1.children[0], IntCst_Impl):
        # f*(n*g) -> (n*f)*g
        return Mult(arg1.children[0] * arg0, arg1.children[1], shapes=shapes)
    elif arg0 == arg1:
        # f*f -> f^2
        return Square(arg0)
    elif isinstance(arg1, SumT_Impl):
        # f*SumT(g)
        if isinstance(arg0, SumT_Impl):
            # SumT(f)*SumT(g) -> SumT(f*g)
            if arg0.dim != arg1.dim:  # should never happen...
                KeOps_Error("dimensions are not compatible for Mult operation")
            return SumT(arg0.children[0] * arg1.children[0], arg0.dim)
        elif arg0.dim == 1:
            # f*SumT(g) -> SumT(f*g)
            return SumT(arg0 * arg1.children[0], arg1.dim)
        elif arg1.dim == arg0.dim:
            # f*SumT(g) -> f*g (broadcasted)
            return arg0 * arg1.children[0]
        else:
            KeOps_Error("dimensions are not compatible for Mult operation")
    elif isinstance(arg0, SumT_Impl):
        # SumT(f)*g -> g*SumT(f)
        return arg1 * arg0
    else:
        return Mult_Impl(arg0, arg1, shapes=shapes)
