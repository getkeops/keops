from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import value
from keopscore.utils.misc_utils import KeOps_Error

############################
######    ELEMENT EXTRACTION : Elem(f,m) (aka get_item)       #####
############################


class Elem(Operation):
    string_id = "Elem"

    def __init__(self, f, m):
        super().__init__(f, params=(m,))
        if f.dim <= m:
            KeOps_Error("Index out of bound in Elem")
        self.dim = 1
        self.m = m

    def Op(self, out, table, arg):
        return value(out).assign(arg[self.m])

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.ElemT import ElemT

        f = self.children[0]
        return f.DiffT(v, ElemT(gradin, f.dim, self.m))

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 1  # number of arguments
    test_argdims = [5]  # dimensions of arguments for testing
    test_params = [3]  # dimensions of parameters for testing
    torch_op = "lambda x, m : x[..., m][..., None]"
