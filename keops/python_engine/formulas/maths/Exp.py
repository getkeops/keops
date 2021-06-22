from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_exp

##########################
######    Exp        #####
##########################

class Exp(VectorizedScalarOp):
    """the exponential vectorized operation"""
    string_id = "Exp"
    
    ScalarOpFun = keops_exp
      
    @property
    def Derivative(self):  
        f = self.children[0]
        return Exp(f)
