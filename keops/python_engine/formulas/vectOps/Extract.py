from keops.python_engine.utils.code_gen_utils import c_array, VectCopy
from keops.python_engine.formulas.Operation import Operation

# //////////////////////////////////////////////////////////////
# ////     VECTOR EXTRACTION : Extract<F,START,DIM>         ////
# //////////////////////////////////////////////////////////////


class Extract(Operation):

    string_id = "Extract"

    def __init__(self, arg0, start, dim):
        if arg0.dim < start + dim or start < 0:
            raise ValueError("Index out of bound in Extract")
        super().__init__(arg0, params=(start, dim))
        self.start = start
        self.dim = dim

    def Op(self, out, table, arg0):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        v = c_array(arg0.dtype, out.dim, f"({arg0.id}+{self.start})")
        return VectCopy(out, v)

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.vectOps.ExtractT import ExtractT

        f = self.children[0]
        return f.DiffT(v, ExtractT(gradin, self.start, f.dim))
