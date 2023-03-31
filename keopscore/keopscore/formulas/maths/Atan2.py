from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_atan2


# //////////////////////////////////////////////////////////////
# ////                 ATAN2 :  Atan2< F, G >               ////
# //////////////////////////////////////////////////////////////


class Atan2(VectorizedScalarOp):
    string_id = "Atan2"

    ScalarOpFun = keops_atan2

    @staticmethod
    def Derivative(f, g):
        r2 = f**2 + g**2
        return g / r2, -f / r2
