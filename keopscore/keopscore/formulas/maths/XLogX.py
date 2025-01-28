from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Log import Log
from keopscore.utils.math_functions import keops_xlogx


class XLogX(VectorizedScalarOp):
    """the x*log(x) vectorized operation"""

    string_id = "XLogX"

    ScalarOpFun = keops_xlogx

    @staticmethod
    def Derivative(f):
        return Log(f) + 1

    # parameters for testing the operation (optional)
    test_ranges = [(0, 2)]  # range of argument

    @staticmethod
    def torch_op():
        """equivalent torch operation"""
        import torch

        return lambda x: x * torch.log(x)
