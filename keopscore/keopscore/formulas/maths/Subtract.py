from keopscore.formulas.Operation import Broadcast
from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.maths.Mult import Mult_Impl
from keopscore.formulas.variables.IntCst import IntCst, IntCst_Impl

##########################
######    Subtract   #####
##########################


class Subtract_Impl(VectorizedScalarOp):
    """the binary subtract operation"""

    string_id = "Subtract"
    print_spec = "-", "mid", 4

    def ScalarOp(self, out, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}-{arg1.id};\n"

    def DiffT(self, v, gradin):
        fa, fb = self.children
        return fa.DiffT(v, gradin) - fb.DiffT(v, gradin)

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments
    torch_op = "torch.sub"  # equivalent PyTorch operation


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Subtract(arg0, arg1):
    if isinstance(arg0, Zero):
        return -Broadcast(arg1, arg0.dim)
    elif isinstance(arg1, Zero):
        return Broadcast(arg0, arg1.dim)
    elif arg0 == arg1:
        return Zero(arg0.dim)
    elif isinstance(arg0, Mult_Impl) and isinstance(arg0.children[0], IntCst_Impl):
        if arg0.children[1] == arg1:
            #  factorization :  n*x - x = (n-1)*x
            return IntCst(arg0.children[0].val - 1) * arg1
        elif (
            isinstance(arg1, Mult_Impl)
            and isinstance(arg1.children[0], IntCst_Impl)
            and arg1.children[1] == arg0.children[1]
        ):
            #  factorization :  m*x - n*x = (m-n)*x
            return (
                IntCst(arg0.children[0].val - arg1.children[0].val) * arg0.children[1]
            )
    if (
        isinstance(arg1, Mult_Impl)
        and isinstance(arg1.children[0], IntCst_Impl)
        and arg1.children[1] == arg0
    ):
        #  factorization :  x - n*x = (1-n)*x
        return IntCst(1 - arg1.children[0].val) * arg0
    else:
        return Subtract_Impl(arg0, arg1)
