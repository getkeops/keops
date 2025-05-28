from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_expn


#####################################
######    Expn functions        #####
#####################################


class Expn(VectorizedScalarOp):
    """the Expn vectorized operation"""

    def __init__(self, f, n=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if n is None:
            # here params should be a tuple containing one single integer
            (n,) = params
        super().__init__(f, params=(n,))
        self.n = n


    string_id = "Expn"

    ScalarOpFun = keops_expn

    @staticmethod
    def Derivative(f):
        from keopscore.formulas import Exp
        Exp(-f) / f