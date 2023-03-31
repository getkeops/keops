from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import c_for_loop
from keopscore.formulas.complex.ComplexReal import ComplexReal
from keopscore.formulas.complex.ComplexMult import ComplexMult
from keopscore.formulas.complex.Conj import Conj
from keopscore.utils.misc_utils import KeOps_Error

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexSquareAbs                           ////
# /////////////////////////////////////////////////////////////////////////


class ComplexSquareAbs(Operation):

    string_id = "ComplexSquareAbs"

    def __init__(self, f):
        if f.dim % 2 != 0:
            KeOps_Error("Dimension of F must be even")
        self.dim = int(f.dim / 2)
        super().__init__(f)

    def Op(self, out, table, inF):
        f = self.children[0]
        forloop, i = c_for_loop(0, f.dim, 2, pragma_unroll=True)
        return forloop(out[i / 2].assign(inF[i] * inF[i] + inF[i + 1] * inF[i + 1]))

    def DiffT(self, v, gradin):
        f = self.children[0]
        AltFormula = ComplexReal(ComplexMult(f, Conj(f)))
        return AltFormula.DiffT(v, gradin)
