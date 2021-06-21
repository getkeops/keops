from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp


##########################
######    Rsqrt      #####
##########################

class Rsqrt(VectorizedScalarOp):
    """the inverse square root vectorized operation"""
    string_id = "Rsqrt"

    def ScalarOp(self, out, arg):
        from keops.python_engine.utils.math_functions import keops_rsqrt
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(keops_rsqrt(arg)) # TODO: check HALF_PRECISION implementation

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.basicMathOps.IntInv import IntInv
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.Grad(v, IntInv(-2) * Rsqrt(f) ** 3 * gradin)
