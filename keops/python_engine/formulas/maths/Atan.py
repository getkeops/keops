
from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_atan

class Atan(VectorizedScalarOp):
    """the arc-sine vectorized operation"""
    string_id = "Atan"

    ScalarOpFun = keops_atan
    
    @property
    def Derivative(self):  
        f = self.children[0]
        return 1/(1+f**2)

