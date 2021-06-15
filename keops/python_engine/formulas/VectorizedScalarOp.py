from keops.python_engine.utils.code_gen_utils import VectApply
from keops.python_engine.formulas.Operation import Operation


class VectorizedScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # scalar operations,
    # such as Exp(f), Cos(f), Mult(f,g), Subtract(f,g), etc.

    def __init__(self, *args):
        dims = set(arg.dim for arg in args)
        if len(dims)>2 or (len(dims)==2 and min(dims)!=1):
            raise ValueError("dimensions are not compatible for VectorizedScalarOp")
        super().__init__(*args)

    @property
    def dim(self):
        # dim gives the output dimension of the operation,
        # here it is the same as the output dimension of the child operation
        return max(child.dim for child in self.children)

    def Op(self, out, table, *arg):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, *arg)