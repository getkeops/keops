from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.utils.code_gen_utils import c_for_loop

#/////////////////////////////////////////////////////////////////////////
#////      ComplexScal                           ////
#/////////////////////////////////////////////////////////////////////////

class ComplexScal_Impl(Operation):
    string_id = "ComplexScal"

    def __init__(self, f, g):
        if f.dim != 2:
            raise ValueError("Dimension of F must be 2")
        if g.dim % 2 != 0:
            raise ValueError("Dimension of G must be even")
        self.dim = g.dim
        super().__init__(f, g)
    
    def Op(self, out, table, inF, inG):
        forloop, i = c_for_loop(0, out.dim, 2, pragma_unroll=True)
        body_str = out[i].assign( inF[0]*inG[i] - inF[1]*inG[i+1] )
        body_str += out[i+1].assign( inF[0]*inG[i+1] + inF[1]*inG[i] )
        return forloop(body_str)

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.complex.ComplexSum import ComplexSum
        from keops.python_engine.formulas.complex.ComplexMult import ComplexMult
        from keops.python_engine.formulas.complex.Conj import Conj
        f, g = self.children
        DiffTF = f.DiffT(v, gradin)
        DiffTG = g.DiffT(v, gradin)
        return Add(DiffTF(v, ComplexSum(ComplexMult(Conj(g), gradin))), DiffTG(v, ComplexMult(Conj(f), gradin)))


def ComplexScal(f,g):
    return ComplexScal_Impl(f,g) if f.dim==2 else ComplexScal_Impl(g,f)


