from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import c_zero_float, c_for_loop

# /////////////////////////////////////////////////////////////////////////
# ////      Real2Complex                           ////
# /////////////////////////////////////////////////////////////////////////


class Real2Complex(Operation):

    string_id = "Real2Complex"

    def __init__(self, f):
        self.dim = 2 * f.dim
        super().__init__(f)

    def Op(self, out, table, inF):
        forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
        body = out[i].assign(inF[i / 2])
        body += out[i + 1].assign(c_zero_float)
        return forloop(body)

    def DiffT(self, v, gradin):
        from keopscore.formulas.complex.ComplexReal import ComplexReal

        f = self.children[0]
        return f.DiffT(v, ComplexReal(gradin))
