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
        return f.DiffT(v, ComplexMult(Conj(g), gradin)) + g.DiffT(
            v, ComplexMult(Conj(f), gradin)
        )
