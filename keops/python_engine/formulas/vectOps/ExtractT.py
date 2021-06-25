from keops.python_engine.utils.code_gen_utils import c_zero_float, VectCopy
from keops.python_engine.formulas.Operation import Operation

# //////////////////////////////////////////////////////////////
# ////     VECTOR "INJECTION" : ExtractT<F,START,DIM>       ////
# //////////////////////////////////////////////////////////////


class ExtractT(Operation):
    string_id = "ExtractT"

    def __init__(self, f, start, dim):
        if start + f.dim > dim or start < 0:
            raise ValueError("Index out of bound in ExtractT")
        super().__init__(f, params=(start, dim))
        self.start = start
        self.dim = dim
        self.dimarg = f.dim

    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        out_prev, out_mid, out_end = out.split(
            self.start, self.dim, self.dimarg - self.start - self.dim
        )
        return "\n".join(
            out_prev.assign(c_zero_float),
            VectCopy(out_mid, arg),
            out_end.assign(c_zero_float),
        )

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.vectOps.Extract import Extract

        f = self.children[0]
        return f.DiffT(v, Extract(gradin, self.start, f.dim))
