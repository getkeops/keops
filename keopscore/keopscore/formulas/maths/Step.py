from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.math_functions import keops_step


class Step(VectorizedScalarOp):
    """the Step vectorized operation"""

    string_id = "Step"

    ScalarOpFun = keops_step

    def DiffT(self, v, gradin):
        return Zero(v.dim)
