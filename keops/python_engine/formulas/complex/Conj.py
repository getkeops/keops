from keops.python_engine.formulas.Operation import Operation

#/////////////////////////////////////////////////////////////////////////
#////      Conj : complex conjugate                           ////
#/////////////////////////////////////////////////////////////////////////

class Conj(Operation):
    string_id = "Conj"

    def __init__(self, f):
        if f.dim % 2 != 0:
            raise ValueError("Dimension of F must be even")
        self.dim = f.dim
        super().__init__(f)
    
    def Op(self, out, table, inF):
        f = self.children[0]
        forloop, i = c_for_loop(0, f.dim, 2, pragma_unroll=True)
        body = out[i].assign( inF[i] )
        body += out[i+1].assign( -inF[i+1] )
        return forloop(body)

    def DiffT(self, v, gradin):
        f = self.children[0]
        return f.DiffT(v, Conj(gradin))
