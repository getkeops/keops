from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.ClampInt import ClampInt
from keopscore.utils.math_functions import keops_clamp


class Clamp(VectorizedScalarOp):
    """Clamp(x,a,b) = a if x<a, x if a<=x<=b, b if b<x"""

    string_id = "Clamp"

    ScalarOpFun = keops_clamp

    def DiffT(self, v, gradin):
        # N.B.   Clamp(F,G,H) = Clamp((F-G)/(H-G),0,1) * (H-G) + G
        # We use this fact to avoid writing another custom operation for the gradient.
        # (This may be slower however...)
        f, g, h = self.children
        Alt_Clamp = ClampInt((f - g) / (h - g), 0, 1) * (h - g) + g
        return Alt_Clamp.DiffT(v, gradin)

    # parameters for testing the operation (optional)
    nargs = 3  # number of arguments
    torch_op = None  # equivalent PyTorch operation
