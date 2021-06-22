from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_rcp

##########################
######    INVERSE : Inv<F>        #####
##########################

class Inv(VectorizedScalarOp):
    """the "Inv" vectorized operation"""
    string_id = "Inv"
    print_spec = "1/", "pre", 2
    
    ScalarOpFun = keops_rcp
    
    @property
    def Derivative(self):  
        f = self.children[0]
        return -1/f**2

