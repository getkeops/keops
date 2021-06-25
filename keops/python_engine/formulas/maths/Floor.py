from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_floor
from keops.python_engine.formulas.variables.Zero import Zero


class Floor(VectorizedScalarOp):

    """the floor vectorized operation"""

    string_id = "Floor"

    ScalarOpFun = keops_floor

    def DiffT(self, v, gradin):
        return Zero(v.dim)
