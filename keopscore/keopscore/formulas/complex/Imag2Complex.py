from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.meta_toolbox import c_zero_float, c_for_loop

# /////////////////////////////////////////////////////////////////////////
# ////      Imag2Complex                           ////
# /////////////////////////////////////////////////////////////////////////


class Imag2Complex(Operation):
    string_id = "Imag2Complex"

    def __init__(self, f):
        self.dim = 2 * f.dim
        super().__init__(f)

    def Op(self, out, table, inF):
        forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
        body = out[i].assign(c_zero_float)
        body += out[i + 1].assign(inF[i / 2])
        return forloop(body)

    def DiffT_fun(self, v, gradin):
        from keopscore.formulas.complex.ComplexImag import ComplexImag

        f = self.children[0]
        return f.DiffT(v, ComplexImag(gradin))
