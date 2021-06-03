from keops.python_engine.formulas.maths.VectorizedScalarOp import VectorizedScalarOp


class Abs(VectorizedScalarOp):
    """the absolute value vectorized operation"""
    string_id = "Abs"

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_abs(arg)};\n"

    def DiffT(self, v, gradin):
        f = self.children[0]
        return f.Grad(v, Sign(f) * gradin)


def keops_abs(x):
    # returns the C++ code string for the abs function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float","double"]:
        return f"abs({x.id})"
    else:
        raise ValueError("not implemented.")