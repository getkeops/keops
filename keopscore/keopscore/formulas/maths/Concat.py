from keopscore.formulas.Operation import Operation
from keopscore.formulas.maths.Extract import Extract
from keopscore.utils.code_gen_utils import VectCopy
from keopscore.utils.misc_utils import KeOps_Error


############################
######    Concat       #####
############################


class Concat(Operation):
    string_id = "Concat"
    print_spec = ("[", "]"), "brackets", 9
    linearity_type = "all"

    def __init__(self, arg0, arg1, params=()):
        # N.B. params keyword is used for compatibility with base class, but should always equal ()
        if params != ():
            KeOps_Error("There should be no parameter.")
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
