from keopscore.formulas import Var, Operation
from keopscore.utils.code_gen_utils import GetInds

class Factorize_Impl(Operation):
    string_id = "Factorize"
    
    def __init__(self, f, g, v):
        super().__init__(f, g, params=(v))
        self.dim = f.dim
        self.f = f
        self.g = g

    def Op(self, out, table, arg):
        return ...

    def DiffT(self, v, gradin):
        return Factorize(f.DiffT(v, gradin), g)

    # parameters for testing the operation (optional)
    enable_test = False  # enable testing for this operation


def Factorize(formula, g):
    inds = GetInds(formula.Vars_)
    newind = 1 + max(inds) if len(inds) > 0 else 0
    v = Var(newind,g.dim,3)
    newformula, cnt = formula.replace_and_count(formula, g, v)
    if cnt>1:
        return Factorize_Impl(newformula,g,v)
    else:
        return formula
    
    
            