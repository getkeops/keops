from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_exp


##########################
######    Exp        #####
##########################


class Exp(VectorizedScalarOp):
    """the exponential vectorized operation"""

    string_id = "Exp"

    ScalarOpFun = keops_exp

    @staticmethod
    def Derivative(f):
        return Exp(f)
