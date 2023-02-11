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
        from keopscore.formulas.complex.ComplexSum import ComplexSum

        f, g = self.children
        if f.dim == 2 and g.dim > 2:
            return ComplexAdd(f.DiffT(v, ComplexSum(gradin)), g.DiffT(v, gradin))
        elif g.dim == 2 and f.dim > 2:
            return ComplexAdd(f.DiffT(v, gradin), g.DiffT(v, ComplexSum(gradin)))
        else:
            return ComplexAdd(f.DiffT(v, gradin), g.DiffT(v, gradin))
