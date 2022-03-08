from keopscore.formulas.VectorizedComplexScalarOp import VectorizedComplexScalarOp

# /////////////////////////////////////////////////////////////////////////
# ////      Conj : complex conjugate                           ////
# /////////////////////////////////////////////////////////////////////////


class Conj(VectorizedComplexScalarOp):

    string_id = "Conj"

    def ScalarOp(self, out, inF):
        string = out[0].assign(inF[0])
        string += out[1].assign(-inF[1])
        return string

    def DiffT(self, v, gradin):
        f = self.children[0]
        return f.DiffT(v, Conj(gradin))
