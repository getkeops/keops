from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Step import Step
from keopscore.utils.math_functions import keops_relu


class ReLU(VectorizedScalarOp):
    """the ReLU vectorized operation"""

    string_id = "ReLU"

    ScalarOpFun = keops_relu

    @staticmethod
    def Derivative(f):
        return Step(f)

    @staticmethod
    def torch_op():
        """equivalent torch operation"""
        import torch

        return torch.relu
