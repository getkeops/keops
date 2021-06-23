from keops.python_engine.utils.code_gen_utils import value, c_zero_float, VectCopy
from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.vectOps.Elem import Elem


############################
######    ELEMENT "INJECTION" : ElemT(f,m,n)
############################

class ElemT(Operation):
    
    string_id = "ElemT"

    def __init__(self, f, m, n):
        super().__init__(f, params=(m,n))
        if f.dim != 1:
            raise ValueError("Input of ElemT should be a scalar")
        self.dim = n
        self.m = m
        self.n = n

    def Op(self, out, table, arg):
        m, n = self.m, self.n
        string = VectCopy(out, c_zero_float, m-1)
        string += out[m].assign(value(arg))
        string += VectCopy(out+m+1, c_zero_float, n-m-1)
        return string

    def DiffT(self, v, gradin):
        f = self.children[0]
        return f.DiffT(v, Elem(gradin, self.m))


