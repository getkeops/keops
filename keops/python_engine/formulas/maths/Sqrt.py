from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_sqrt
from keops.python_engine.formulas.maths.Rsqrt import Rsqrt
from keops.python_engine.formulas.basicMathOps.IntInv import IntInv  
      
##########################
######    Sqrt       #####
##########################

class Sqrt(VectorizedScalarOp):
    
    """the square root vectorized operation"""
    
    string_id = "Sqrt"

    ScalarOpFun = keops_sqrt
    
    @staticmethod
    def Derivative(f):  
        return IntInv(2) * Rsqrt(f)

