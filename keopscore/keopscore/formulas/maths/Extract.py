from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import c_array, VectCopy
from keopscore.utils.misc_utils import KeOps_Error

# //////////////////////////////////////////////////////////////
# ////     VECTOR EXTRACTION : Extract<F,START,DIM>         ////
# //////////////////////////////////////////////////////////////


class Extract(Operation):
    string_id = "Extract"

    def __init__(self, arg0, start, dim):
        if arg0.dim < start + dim or start < 0:
            KeOps_Error("Index out of bound in Extract")
        super().__init__(arg0, params=(start, dim))
        self.start = start
        self.dim = dim

    def Op(self, out, table, arg0):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        v = c_array(arg0.dtype, out.dim, f"({arg0.id}+{self.start})")
        return VectCopy(out, v)

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.ExtractT import ExtractT

        f = self.children[0]
        return f.DiffT(v, ExtractT(gradin, self.start, f.dim))

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 1  # number of arguments
    test_argdims = [10]  # dimensions of arguments for testing
    test_params = [3, 5]  # values of parameters for testing
    torch_op = "lambda x,s,d : x[...,s:(s+d)]"  # equivalent PyTorch operation
