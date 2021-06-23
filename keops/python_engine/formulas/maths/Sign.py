from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_sign
from keops.python_engine.formulas.variables.Zero import Zero

##########################
######    Sign        #####
##########################

class Sign(VectorizedScalarOp):
    """the sign vectorized operation"""
    string_id = "Sign"
    
    ScalarOpFun = keops_sign
      
    def DiffT(self, v, gradin):
        return Zero(v.dim)
