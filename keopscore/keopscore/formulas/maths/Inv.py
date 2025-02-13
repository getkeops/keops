from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_rcp


##########################
######    INVERSE : Inv<F>        #####
##########################


class Inv(VectorizedScalarOp):
    """the "Inv" vectorized operation"""

    string_id = "Inv"
    print_spec = "1/", "pre", 3

    ScalarOpFun = keops_rcp

    @staticmethod
    def Derivative(f):
        return -1 / f**2

    # parameters for testing the operation (optional)
    @staticmethod
    def torch_op():
        """equivalent torch operation"""
        import torch

        return lambda x: 1 / x
