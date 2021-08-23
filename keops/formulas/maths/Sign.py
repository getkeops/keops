from keops.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.formulas.variables.Zero import Zero
from keops.utils.math_functions import keops_sign


##########################
######    Sign        #####
##########################


class Sign(VectorizedScalarOp):
    """the sign vectorized operation"""

    string_id = "Sign"

    ScalarOpFun = keops_sign

    def DiffT(self, v, gradin):
        return Zero(v.dim)
