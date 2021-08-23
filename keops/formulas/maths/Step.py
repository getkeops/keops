from keops.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.formulas.variables.Zero import Zero
from keops.utils.math_functions import keops_step


class Step(VectorizedScalarOp):
    """the Step vectorized operation"""

    string_id = "Step"

    ScalarOpFun = keops_step

    def DiffT(self, v, gradin):
        return Zero(v.dim)
