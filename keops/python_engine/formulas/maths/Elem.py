from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.utils.code_gen_utils import value


############################
######    ELEMENT EXTRACTION : Elem(f,m) (aka get_item)       #####
############################


class Elem(Operation):
    string_id = "Elem"

    def __init__(self, f, m):
        super().__init__(f, params=(m,))
        if f.dim <= m:
            raise ValueError("Index out of bound in Elem")
        self.dim = 1
        self.m = m

    def Op(self, out, table, arg):
        return value(out).assign(arg[self.m])

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.maths.ElemT import ElemT

        f = self.children[0]
        return f.DiffT(v, ElemT(gradin, f.dim, self.m))
