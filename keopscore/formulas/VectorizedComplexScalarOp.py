from keopscore.utils.code_gen_utils import ComplexVectApply
from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error


class VectorizedComplexScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # complex scalar operations,
    # such as ComplexExp(f), ComplexMult(f), ComplexAdd(f,g), etc.

    def __init__(self, *args, params=()):
        dims = set(arg.dim for arg in args)
        if max(dims) % 2 != 0 or len(dims) > 2 or (len(dims) == 2 and min(dims) != 2):
            KeOps_Error("dimensions are not compatible for VectorizedComplexScalarOp")
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
