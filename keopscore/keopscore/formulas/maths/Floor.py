from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.math_functions import keops_floor


class Floor(VectorizedScalarOp):
    """the floor vectorized operation"""

    string_id = "Floor"

    ScalarOpFun = keops_floor

    def DiffT(self, v, gradin):
        return Zero(v.dim)
