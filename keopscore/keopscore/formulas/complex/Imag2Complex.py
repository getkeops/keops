from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import c_zero_float, c_for_loop

# /////////////////////////////////////////////////////////////////////////
# ////      Imag2Complex                           ////
# /////////////////////////////////////////////////////////////////////////


class Imag2Complex(Operation):
    string_id = "Imag2Complex"

    def __init__(self, f, params=()):
        # N.B. params keyword is used for compatibility with base class, but should always equal ()
        if params != ():
            KeOps_Error("There should be no parameter.")
        self.dim = 2 * f.dim
        super().__init__(f)

    def Op(self, out, table, inF):
        forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
        body = out[i].assign(c_zero_float)
        body += out[i + 1].assign(inF[i / 2])
        return forloop(body)

    def DiffT(self, v, gradin):
        from keopscore.formulas.complex.ComplexImag import ComplexImag

        f = self.children[0]
        return f.DiffT(v, ComplexImag(gradin))
