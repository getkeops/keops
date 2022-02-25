from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.math_functions import keops_sign


##########################
######    Sign        #####
##########################


class Sign(VectorizedScalarOp):
    """the sign vectorized operation"""

    string_id = "Sign"

    ScalarOpFun = keops_sign

    def DiffT(self, v, gradin):
        return Zero(v.dim)
