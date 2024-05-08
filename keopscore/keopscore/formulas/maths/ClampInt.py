from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.DiffClampInt import DiffClampInt
from keopscore.utils.math_functions import keops_clampint
from keopscore.utils.unique_object import unique_object


def ClampInt(x, a, b):
    return ClampInt_Impl_Factory(a, b)(x)


class ClampInt_Impl(VectorizedScalarOp):

    # parameters for testing the operation (optional)
    test_params = [0, 1]  # parameters to try
    torch_op = "torch.clamp"  # equivalent PyTorch operation


class ClampInt_Impl_Factory(metaclass=unique_object):

    def __init__(self, a, b):

        class Class(ClampInt_Impl):
            """ClampInt(x,a,b) = a if x<a, x if a<=x<=b, b if b<x
            N.B. same as Clamp but a and b are fixed integers.
            ClampInt may be faster than Clamp because we avoid the transfer
            of A and B in memory.
            """

            string_id = "ClampInt"
            print_fun = lambda x: f"ClampInt({x},{a},{b})"

            ScalarOpFun = lambda x: keops_clampint(x, a, b)

            @staticmethod
            def Derivative(x):
                return DiffClampInt(x, a, b)

        self.Class = Class

    def __call__(self, f):
        return self.Class(f)
