from keopscore.formulas.Operation import Broadcast
from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Scalprod import Scalprod
from keopscore.formulas.maths.Sum import Sum
from keopscore.formulas.maths.Square import Square
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.misc_utils import KeOps_Error

##########################
######    Divide     #####
##########################


class Divide_Impl(VectorizedScalarOp):
    """the binary divide operation"""

    string_id = "Divide"
    print_spec = "/", "mid", 3

    def ScalarOp(self, out, arg0, arg1):
        """returns the atomic piece of c++ code to evaluate the function on arg and return the result in out"""
        return f"{out.id} = {arg0.id} / {arg1.id};\n"
  
    @staticmethod
    def Derivative(a, b):
        return 1/b, -a/b**2

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Divide(arg0, arg1, shapes=None):
    if isinstance(arg0, Zero):
        return Broadcast(arg0, arg1.dim)
    elif isinstance(arg1, Zero):
        KeOps_Error("division by zero")
    else:
        return Divide_Impl(arg0, arg1, shapes=shapes)
