from keops.utils.code_gen_utils import ComplexVectApply
from keops.formulas.Operation import Operation


class VectorizedComplexScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # complex scalar operations,
    # such as ComplexExp(f), ComplexMult(f), ComplexAdd(f,g), etc.

    def __init__(self, *args, params=()):
        dims = set(arg.dim for arg in args)
        if max(dims) % 2 != 0 or len(dims) > 2 or (len(dims) == 2 and min(dims) != 2):
            raise ValueError(
                "dimensions are not compatible for VectorizedComplexScalarOp"
            )
        super().__init__(*args, params=params)

    @property
    def dim(self):
        # dim gives the output dimension of the operation,
        # here it is the same as the output dimension of the child operation
        return max(child.dim for child in self.children)

    def Op(self, out, table, *args):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return ComplexVectApply(self.ScalarOp, out, *args)
