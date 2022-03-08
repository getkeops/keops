from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_diffclampint

"""
//////////////////////////////////////////////////////////////
////         DIFFCLAMPINT : DiffClampInt< F, A, B >       ////
//////////////////////////////////////////////////////////////

// DiffClampInt(x,a,b) = 0 if x<a, 1 if a<=x<=b, 0 if b<x 
// N.B. used as derivative of ClampInt operation
"""


class DiffClampInt(VectorizedScalarOp):

    string_id = "DiffClampInt"

    def __init__(self, x, a, b):
        super().__init__(x, params=(a, b))

    ScalarOpFun = keops_diffclampint

    def DiffT(self, v, gradin):
        from keopscore.formulas import Zero

        return Zero(v.dim)

    # parameters for testing the operation (optional)
    enable_test = (
        False  # (because it will be tested anyway if we test the gradient of ClampInt)
    )
