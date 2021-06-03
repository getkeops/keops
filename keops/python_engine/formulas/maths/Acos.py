from keops.python_engine.formulas.maths.Rsqrt import Rsqrt
from keops.python_engine.formulas.maths.Square import Square
from keops.python_engine.formulas.maths.VectorizedScalarOp import VectorizedScalarOp


class Acos(VectorizedScalarOp):
    """the arc-cosine vectorized operation"""
    string_id = "Acos"

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {self.keops_acos(arg)};\n"

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.variables.IntCst import IntCst
        f = self.children[0]
        return f.Grad(v, - Rsqrt((IntCst(1) - Square(f))) * gradin)

    @staticmethod
    def keops_acos(x):
        # returns the C++ code string for the acos function applied to a C++ variable
        # - x must be of type c_variable
        if x.dtype in ["float", "double"]:  # TODO: check CUDA_ARCH version
            return f"acos({x.id})"
        else:
            raise ValueError("not implemented.")
