from keops.python_engine.code_gen_utils import VectCopy
from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.vectOps.Extract import Extract


############################
######    Concat       #####
############################

class Concat(Operation):
    string_id = "Concat"

    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim + arg1.dim

    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        out0, out1 = out.split(arg0.dim, arg1.dim)
        return VectCopy(out0, arg0) + VectCopy(out1, arg1)

    def DiffT(self, v, gradin):
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v, Extract(gradin, 0, f.dim)) + g.Grad(v, Extract(gradin, f.dim, g.dim))