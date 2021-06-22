from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_ifelse

class IfElse(VectorizedScalarOp):

    """the if/else vectorized operation
    IfElse(f,a,b) = a if f>=0, b otherwise
    """

    string_id = "IfElse"
    
    ScalarOpFun = keops_ifelse
    
    def DiffT(self, v, gradin):
        f, g, h = self.children
        return IfElse(f, g.DiffT(v,gradin), h.DiffT(v,gradin))
