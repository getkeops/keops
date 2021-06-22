from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp

class ClampInt(VectorizedScalarOp):
    """ ClampInt(x,a,b) = a if x<a, x if a<=x<=b, b if b<x 
        N.B. same as Clamp but a and b are fixed integers.
        ClampInt may be faster than Clamp because we avoid the transfer
        of A and B in memory.
    """
    string_id = "ClampInt"
    
    def __init__(self, x, a, b):
        super().__init__(x)
        self.a = a
        self.b = b

    def ScalarOp(self, out, arg):
        from keops.python_engine.utils.math_functions import keops_clampint
        return out.assign(keops_clampint(arg, self.a, self.b))
    
    @property
    def Derivative(self):  
        from keops.python_engine.formulas.maths.DiffClampInt import DiffClampInt
        x, a, b = self.children[0], self.a, self.b
        return DiffClampInt(x,a,b)
