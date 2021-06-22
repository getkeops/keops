
from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_asin

class Asin(VectorizedScalarOp):
    """the arc-sine vectorized operation"""
    string_id = "Asin"

    ScalarOpFun = keops_asin
        
    @property
    def Derivative(self):  
        from keops.python_engine.formulas.maths.Rsqrt import Rsqrt
        f = self.children[0]
        return Rsqrt(1-f**2)
