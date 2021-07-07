from keops.python_engine.utils.code_gen_utils import value, c_zero_float, c_for_loop
from keops.python_engine.formulas.Operation import Operation


############################
######    ELEMENT "INJECTION" : ElemT(f,m,n)
############################


class ElemT(Operation):

    string_id = "ElemT"

    def __init__(self, f, n, m):
        super().__init__(f, params=(n, m))
        if f.dim != 1:
            raise ValueError("Input of ElemT should be a scalar")
        self.dim = n
        self.n = n
        self.m = m

    def Op(self, out, table, arg):
        n, m = self.n, self.m
        loop1, k = c_for_loop(0, m, 1, pragma_unroll=True)
        string = loop1(out[k].assign(c_zero_float))
        string += out[m].assign(value(arg))
        loop2, k = c_for_loop(m + 1, n, 1, pragma_unroll=True)
        string += loop2(out[k].assign(c_zero_float))
        return string

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.vectOps.Elem import Elem

        f = self.children[0]
        return f.DiffT(v, Elem(gradin, self.m))
