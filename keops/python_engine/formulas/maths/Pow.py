from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_pow

class Pow(VectorizedScalarOp):
    """the integer power vectorized operation
    Pow(f,m) where m is integer, computes f^m
    """

    def __init__(self, f, m):
        super().__init__(f, params=(m,))

    string_id = "Pow"

    ScalarOpFun = keops_pow

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.variables.IntCst import IntCst
        m = self.params[0]
        return IntCst(m)*Pow(f,m-1)






