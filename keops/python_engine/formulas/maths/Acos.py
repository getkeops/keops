
from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp

class Acos(VectorizedScalarOp):
    """the arc-cosine vectorized operation"""
    string_id = "Acos"

    def ScalarOp(self, out, arg):
        from keops.python_engine.utils.math_functions import keops_acos
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(keops_acos(arg))

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.maths.Rsqrt import Rsqrt
        from keops.python_engine.formulas.maths.Square import Square
        from keops.python_engine.formulas.variables.IntCst import IntCst
        f = self.children[0]
        return f.Grad(v, - Rsqrt((IntCst(1) - Square(f))) * gradin)
