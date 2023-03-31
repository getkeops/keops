from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.IntInv import IntInv


##########################
######    Rsqrt      #####
##########################


class Rsqrt(VectorizedScalarOp):
    """the inverse square root vectorized operation"""

    string_id = "Rsqrt"

    def ScalarOp(self, out, arg):
        from keopscore.utils.math_functions import keops_rsqrt

        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(keops_rsqrt(arg))

    @staticmethod
    def Derivative(f):
        return IntInv(-2) * Rsqrt(f) ** 3

    # parameters for testing the operation (optional)
    test_ranges = [(0.5, 2)]  # range of argument
