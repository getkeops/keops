from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox import c_array_from_address
from keopscore.utils.misc_utils import KeOps_Error

# //////////////////////////////////////////////////////////////
# ////     VECTOR EXTRACTION : Extract<F,START,DIM>         ////
# //////////////////////////////////////////////////////////////


def Extract(x, start, dim):
    return Extract_Impl_Factory(start, dim)(x)


class Extract_Impl(Operation):

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 1  # number of arguments
    test_argdims = [10]  # dimensions of arguments for testing
    test_params = [3, 5]  # values of parameters for testing
    torch_op = "lambda x,s,d : x[...,s:(s+d)]"  # equivalent PyTorch operation


class Extract_Impl_Factory:

    def __init__(self, start, dim):

        class Class(Extract_Impl):
            string_id = "Extract"
            linearity_type = "all"

            def __init__(self, arg0):
                if arg0.dim < start + dim or start < 0:
                    KeOps_Error("Index out of bound in Extract")
                super().__init__(arg0)
                self.start = start
                self.dim = dim

            def Op(self, out, table, arg0):
                # returns the atomic piece of c++ code to evaluate the function on arg and return
                # the result in out
                v = c_array_from_address(out.dim, arg0.c_address + self.start)
                return out.copy(v)

            def DiffT(self, v, gradin):
                from keopscore.formulas.maths.ExtractT import ExtractT

                f = self.children[0]
                return f.DiffT(v, ExtractT(gradin, self.start, f.dim))

        self.Class = Class

    def __call__(self, x):
        return self.Class(x)
