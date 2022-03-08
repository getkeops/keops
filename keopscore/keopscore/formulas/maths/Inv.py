from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_rcp


##########################
######    INVERSE : Inv<F>        #####
##########################


class Inv(VectorizedScalarOp):
    """the "Inv" vectorized operation"""

    string_id = "Inv"
    print_spec = "1/", "pre", 2

    ScalarOpFun = keops_rcp

    @staticmethod
    def Derivative(f):
        return -1 / f**2

    # parameters for testing the operation (optional)
    torch_op = "lambda x : 1 / x"
