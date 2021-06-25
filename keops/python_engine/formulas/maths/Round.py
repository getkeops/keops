from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_round
from keops.python_engine.formulas.variables.Zero import Zero


class Round(VectorizedScalarOp):

    """the Round vectorized operation
        Round(f,d) where d is integer, rounds f to d decimals
    """

    def __init__(self, f, d):
        super().__init__(f, params=(d,))

    string_id = "Round"

    ScalarOpFun = keops_round

    def DiffT(self, v, gradin):
        return Zero(v.dim)
