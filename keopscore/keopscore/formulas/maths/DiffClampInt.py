from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_diffclampint

"""
//////////////////////////////////////////////////////////////
////         DIFFCLAMPINT : DiffClampInt< F, A, B >       ////
//////////////////////////////////////////////////////////////

// DiffClampInt(x,a,b) = 0 if x<a, 1 if a<=x<=b, 0 if b<x 
// N.B. used as derivative of ClampInt operation
"""


def DiffClampInt(x, a, b):
    return DiffClampInt_Impl_Factory(a, b)(x)


class DiffClampInt_Impl(VectorizedScalarOp):

    # parameters for testing the operation (optional)
    enable_test = (
        False  # (because it will be tested anyway if we test the gradient of ClampInt)
    )


class DiffClampInt_Impl_Factory:

    def __init__(self, a, b):

        class Class(DiffClampInt_Impl):
            string_id = "DiffClampInt"
            print_fun = lambda x: f"DiffClampInt({x},{a},{b})"

            ScalarOpFun = lambda x : keops_diffclampint(x,a,b)

            def DiffT_fun(self, v, gradin):
                from keopscore.formulas import Zero

                return Zero(v.dim)

        self.Class = Class

    def __call__(self, x):
        return self.Class(x)
