from keopscore.formulas.VectorizedComplexScalarOp import VectorizedComplexScalarOp

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexAdd                           ////
# /////////////////////////////////////////////////////////////////////////


class ComplexAdd(VectorizedComplexScalarOp):

    string_id = "ComplexAdd"

    def ScalarOp(self, out, inF, inG):
        string = out[0].assign(inF[0] + inG[0])
        string += out[1].assign(inF[1] + inG[1])
        return string

    def DiffT(self, v, gradin):
        f, g = self.children
        return ComplexAdd(f.DiffT(v, gradin), g.DiffT(v, gradin))
