from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_expInt


#################################################
######    Expi: Exponential integral        #####
#################################################


class Expi(VectorizedScalarOp):
    """the exponential vectorized operation"""

    string_id = "Expi"

    ScalarOpFun = keops_expInt

    @staticmethod
    def Derivative(f):
        from keopscore.formulas import Exp
        Exp(-f) / f