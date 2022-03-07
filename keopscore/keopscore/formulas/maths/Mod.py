from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_mod


class Mod(VectorizedScalarOp):
    """the Modulo vectorized operation
    Mod(x,n,d) = x - n * Floor((x - d)/n)
    """

    string_id = "Mod"

    ScalarOpFun = keops_mod

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.Floor import Floor

        # we fall back to an alternative definition of Mod for defining the gradient
        x, n, d = self.children
        Mod_alt = x - n * Floor((x - d) / n)
        return Mod_alt.DiffT(v, gradin)

    # parameters for testing the operation (optional)
    nargs = 3  # number of arguments


"""
# N.B. below is alternative definition as a simple alias.
# It is theoretically less efficient when applied to vectors
# because we compose operations so evalauation implies the creation
# of useless temporary vectors. 
# If we implement a sort of "fusion" of vectorized scalar operations,
# it should be completely equivalent.
        
from keopscore.formulas.maths.Floor import Floor
def Mod(x,n,d):
    return x - n * Floor((x - d)/n)
"""
