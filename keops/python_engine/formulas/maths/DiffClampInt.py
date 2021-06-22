from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_diffclampint

"""
//////////////////////////////////////////////////////////////
////         DIFFCLAMPINT : DiffClampInt< F, A, B >       ////
//////////////////////////////////////////////////////////////

// DiffClampInt(x,a,b) = 0 if x<a, 1 if a<=x<=b, 0 if b<x 
// N.B. used as derivative of ClampInt operation
"""

class DiffClampInt(VectorizedScalarOp):

    string_id = "DiffClampInt"
    
    def __init__(self, x, a, b):
        super().__init__(x)
        self.a = a
        self.b = b

    def ScalarOp(self, out, arg):
        return out.assign(keops_diffclampint(arg, self.a, self.b))
    
    def DiffT(self, v, gradin):
        return Zero(v.dim)
