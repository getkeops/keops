from keopscore.formulas.VectorizedComplexScalarOp import VectorizedComplexScalarOp
from keopscore.formulas.complex.Conj import Conj
from keopscore.formulas.complex.ComplexAdd import ComplexAdd

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
        from keopscore.formulas.complex.ComplexSum import ComplexSum

        f, g = self.children
        if f.dim == 2 and g.dim > 2:
            return ComplexAdd(
                f.DiffT(v, ComplexSum(ComplexMult(Conj(g), gradin))),
                g.DiffT(v, ComplexMult(Conj(f), gradin)),
            )
        elif g.dim == 2 and f.dim > 2:
            return ComplexAdd(
                f.DiffT(v, ComplexMult(Conj(g), gradin)),
                g.DiffT(v, ComplexSum(ComplexMult(Conj(f), gradin))),
            )
        else:
            return ComplexAdd(
                f.DiffT(v, ComplexMult(Conj(g), gradin)),
                g.DiffT(v, ComplexMult(Conj(f), gradin)),
            )
