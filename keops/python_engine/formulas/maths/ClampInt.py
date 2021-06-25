from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.formulas.maths.DiffClampInt import DiffClampInt
from keops.python_engine.utils.math_functions import keops_clampint


class ClampInt(VectorizedScalarOp):
    """ ClampInt(x,a,b) = a if x<a, x if a<=x<=b, b if b<x 
        N.B. same as Clamp but a and b are fixed integers.
        ClampInt may be faster than Clamp because we avoid the transfer
        of A and B in memory.
    """

    string_id = "ClampInt"

    def __init__(self, x, a, b):
        super().__init__(x, params=(a, b))

    ScalarOpFun = keops_clampint

    @staticmethod
    def Derivative(x, a, b):
        return DiffClampInt(x, a, b)
