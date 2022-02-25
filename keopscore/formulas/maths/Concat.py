from keopscore.formulas.Operation import Operation
from keopscore.formulas.maths.Extract import Extract
from keopscore.utils.code_gen_utils import VectCopy


############################
######    Concat       #####
############################


class Concat(Operation):
    string_id = "Concat"

    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim + arg1.dim

    def Op(self, out, table, arg0, arg1):
        out0, out1 = out.split(arg0.dim, arg1.dim)
        return VectCopy(out0, arg0) + VectCopy(out1, arg1)

    def DiffT(self, v, gradin):
        f = self.children[0]
        g = self.children[1]
        return f.DiffT(v, Extract(gradin, 0, f.dim)) + g.DiffT(
            v, Extract(gradin, f.dim, g.dim)
        )

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 2  # number of arguments
    test_argdims = [5, 3]  # dimensions of arguments for testing
    torch_op = None
