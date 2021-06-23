from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_step
from keops.python_engine.formulas.variables.Zero import Zero

class Step(VectorizedScalarOp):

    """the Step vectorized operation"""

    string_id = "Step"
    
    ScalarOpFun = keops_step
    
    def DiffT(self, v, gradin):
        return Zero(v.dim)
