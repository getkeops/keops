from keops.python_engine.formulas.maths.VectorizedScalarOp import VectorizedScalarOp


##########################
######    Rsqrt      #####
##########################

class Rsqrt(VectorizedScalarOp):
    """the inverse square root vectorized operation"""
    string_id = "Rsqrt"

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_rsqrt(arg)};\n" # TODO: check HALF_PRECISION implementation

    def DiffT(self, v, gradin):
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.Grad(v, IntInv(-2) * Rsqrt(f) ** 3 * gradin)


def keops_rsqrt():
    return NotImplementedError()