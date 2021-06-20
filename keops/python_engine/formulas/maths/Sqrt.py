from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
        
        
##########################
######    Sqrt       #####
##########################

class Sqrt(VectorizedScalarOp):
    """the square root vectorized operation"""
    string_id = "Sqrt"

    def ScalarOp(self, out, arg):
        from keops.python_engine.utils.math_functions import keops_sqrt
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(keops_sqrt(arg))

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.maths.Rsqrt import Rsqrt
        from keops.python_engine.formulas.basicMathOps.IntInv import IntInv
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.Grad(v, IntInv(2) * Rsqrt(f) * gradin)

