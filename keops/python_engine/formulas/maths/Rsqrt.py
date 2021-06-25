from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.formulas.basicMathOps.IntInv import IntInv

##########################
######    Rsqrt      #####
##########################


class Rsqrt(VectorizedScalarOp):
    """the inverse square root vectorized operation"""

    string_id = "Rsqrt"

    def ScalarOp(self, out, arg):
        from keops.python_engine.utils.math_functions import keops_rsqrt

        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(keops_rsqrt(arg))  # TODO: check HALF_PRECISION implementation

    @staticmethod
    def Derivative(f):
        return IntInv(-2) * Rsqrt(f) ** 3
