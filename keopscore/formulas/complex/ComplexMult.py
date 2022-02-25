from keopscore.formulas.VectorizedComplexScalarOp import VectorizedComplexScalarOp
from keopscore.formulas.complex.Conj import Conj

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexMult                           ////
# /////////////////////////////////////////////////////////////////////////


class ComplexMult(VectorizedComplexScalarOp):

    string_id = "ComplexMult"

    def ScalarOp(self, out, inF, inG):
        string = out[0].assign(inF[0] * inG[0] - inF[1] * inG[1])
        string += out[1].assign(inF[0] * inG[1] + inF[1] * inG[0])
        return string

    def DiffT(self, v, gradin):
        f, g = self.children
        DiffTF = f.DiffT(v, gradin)
        DiffTG = g.DiffT(v, gradin)
        return DiffTF(v, ComplexMult(Conj(g), gradin)) + DiffTG(
            v, ComplexMult(Conj(f), gradin)
        )
