from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.math_functions import keops_round
from keopscore.utils.unique_object import unique_object


def Round(f, d):
    return Round_Impl_Factory(d)(f)


class Round_Impl(VectorizedScalarOp):

    # parameters for testing the operation (optional)
    nargs = 1  # number of arguments
    test_params = [3]  # parameters to try
    torch_op = None  # equivalent PyTorch operation


class Round_Impl_Factory(metaclass=unique_object):

    def __init__(self, d):

        class Class(Round_Impl):
            """the Round vectorized operation
            Round(f,d) where d is integer, rounds f to d decimals
            """

            def __init__(self, f):
                super().__init__(f)

            string_id = "Round"

            ScalarOpFun = keops_round

            def DiffT(self, v, gradin):
                return Zero(v.dim)

        self.Class = Class

    def __call__(self, f):
        return self.Class(f)
