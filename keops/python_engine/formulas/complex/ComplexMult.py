from keops.python_engine.formulas.Operation import Operation

#/////////////////////////////////////////////////////////////////////////
#////      ComplexMult                           ////
#/////////////////////////////////////////////////////////////////////////

class ComplexMult(Operation):
    string_id = "ComplexMult"

    def __init__(self, f, g):
        if f.dim % 2 != 0:
            raise ValueError("Dimension of F must be even")
        if f.dim != g.dim:
            raise ValueError("Dimensions of F and G must be equal")
        self.dim = f.dim
        super().__init__(f, g)
    
    def Op(self, out, table, inF, inG):
        forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
        body = out[i].assign( inF[i]*inG[i] - inF[i+1]*inG[i+1] )
        body += out[i+1].assign( inF[i]*inG[i+1] + inF[i+1]*inG[i] )
        return forloop(body)

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.complex.ComplexMult import ComplexMult
        from keops.python_engine.formulas.complex.Conj import Conj
        from keops.python_engine.formulas.basicMathOps.Add import Add
        f, g = self.children
        DiffTF = f.DiffT(v, gradin)
        DiffTG = g.DiffT(v, gradin)
        return Add(DiffTF(v, ComplexMult(Conj(g), gradin)), DiffTG(v, ComplexMult(Conj(f), gradin)))
    
