from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp

class Abs(VectorizedScalarOp):
    """the absolute value vectorized operation"""
    string_id = "Abs"

    def ScalarOp(self, out, arg):
        from keops.python_engine.utils.math_functions import keops_abs
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(keops_abs(arg))
    
    @property
    def Derivative(self):  
        from keops.python_engine.formulas.maths.Sign import Sign
        f = self.children[0]
        return Sign(f)

    