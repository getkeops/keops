from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.utils.code_gen_utils import c_for_loop
from keops.python_engine.formulas.complex.Real2Complex import Real2Complex
from keops.python_engine.formulas.complex.ComplexScal import ComplexScal
        
#/////////////////////////////////////////////////////////////////////////
#////      ComplexRealScal                           ////
#/////////////////////////////////////////////////////////////////////////

class ComplexRealScal_Impl(Operation):
    
    string_id = "ComplexRealScal"

    def __init__(self, f, g):
        if f.dim != 1:
            raise ValueError("Dimension of F must be 1")
        if g.dim % 2 != 0:
            raise ValueError("Dimension of G must be even")
        self.dim = g.dim
        super().__init__(f, g)
    
    def Op(self, out, table, inF, inG):
        forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
        body = out[i].assign( inF[0]*inG[i] )
        body += out[i+1].assign( inF[0]*inG[i+1] )
        return forloop(body)

    def DiffT(self, v, gradin):
        f, g = self.children
        AltFormula = ComplexScal(Real2Complex(f), g)
        return AltFormula.DiffT(v, gradin)
        

def ComplexRealScal(f,g):
    return ComplexRealScal_Impl(f,g) if f.dim==1 else ComplexRealScal_Impl(g,f)
