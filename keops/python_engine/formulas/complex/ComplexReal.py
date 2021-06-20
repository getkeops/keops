from keops.python_engine.formulas.Operation import Operation

#/////////////////////////////////////////////////////////////////////////
#////      ComplexReal                           ////
#/////////////////////////////////////////////////////////////////////////

class ComplexReal(Operation):
    string_id = "ComplexReal"

    def __init__(self, f):
        if f.dim % 2 != 0:
            raise ValueError("Dimension of F must be even")
        self.dim = int(f.dim / 2)
        super().__init__(f)
    
    def Op(self, out, table, inF):
        f = self.children[0]
        forloop, i = c_for_loop(0, f.dim, 2, pragma_unroll=True)
        return forloop( out[i/2].assign(inF[i]) )

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.complex.Real2Complex import Real2Complex
        f = self.children[0]
        return f.DiffT(v, Real2Complex(gradin))


