from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
import keops.python_engine.formulas.maths.Rsqrt as Rsqrt

##########################
######    Sqrt       #####
##########################

class Sqrt(VectorizedScalarOp):
    """the square root vectorized operation"""
    string_id = "Sqrt"

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_sqrt(arg)};\n"

    def DiffT(self, v, gradin):
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.Grad(v, IntInv(2) * Rsqrt(f) * gradin)


def keops_sqrt(x):
    # returns the C++ code string for the square root function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float","double"]:
        return f"sqrt({x.id})"
    else:
        raise ValueError("not implemented.")