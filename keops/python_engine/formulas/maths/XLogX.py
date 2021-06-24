from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_xlogx
from keops.python_engine.formulas.maths.Log import Log


class XLogX(VectorizedScalarOp):
    
    """the x*log(x) vectorized operation"""
    
    string_id = "XLogX"

    ScalarOpFun = keops_xlogx
    
    @staticmethod
    def Derivative(f):  
        return Log(f)+1




    test_ranges = [(0,2)]