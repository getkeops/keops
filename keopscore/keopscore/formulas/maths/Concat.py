from keopscore.formulas.Operation import Operation
from keopscore.formulas.maths.Extract import Extract
from keopscore.utils.misc_utils import KeOps_Error


############################
######    Concat       #####
############################


class Concat(Operation):
    string_id = "Concat"
    print_fun = lambda x, y: f"[{x},{y}]"
    print_level = 9
    linearity_type = "all"

    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim + arg1.dim

    def Op(self, out, table, arg0, arg1):
        out0, out1 = out.split(arg0.dim, arg1.dim)
        return out0.copy(arg0) + out1.copy(arg1)

    def DiffT_fun(self, v, gradin):
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
