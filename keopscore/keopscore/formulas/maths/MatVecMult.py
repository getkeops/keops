from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import (
    c_variable,
    c_for_loop,
    c_zero_float,
)
from keopscore.utils.misc_utils import KeOps_Error

# /////////////////////////////////////////////////////////////////////////
# ////     Matrix-vector product      A x b                           ////
# /////////////////////////////////////////////////////////////////////////


class MatVecMult(Operation):
    string_id = "MatVecMult"

    def __init__(self, A, B):
        # A is vector of size n*p, interpreted as matrix, B is vector of size p, interpreted as column vector
        # output is vector of size n
        if A.dim % B.dim != 0:
            KeOps_Error(
                "Dimensions of A and B are not compatible for matrix-vector product"
            )
        super().__init__(A, B)
        self.dim = A.dim // B.dim

    def Op(self, out, table, inA, inB):
        q = c_variable("int")
        loop, i = c_for_loop(0, self.dim, 1, pragma_unroll=True)
        inner_loop, k = c_for_loop(0, inB.dim, 1, pragma_unroll=True)
        return f"""
                    #if C_CONTIGUOUS     // row major
                        {q.declare_assign(0)}
                        {loop(out[i].assign(c_zero_float) + inner_loop(out[i].add_assign(inA[q] * inB[k]) + q.add_assign(1)))}
                    #else               // column major
                        {loop(out[i].assign(c_zero_float) + inner_loop(out[i].add_assign(inA[k * self.dim + i] * inB[k])))}
                    #endif
                """

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths import TensorProd, VecMatMult

        A, B = self.children
        return A.DiffT(v, TensorProd(gradin, B)) + B.DiffT(v, VecMatMult(gradin, A))

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 2  # number of arguments
    test_argdims = [6, 2]  # dimensions of arguments for testing
    torch_op = None  # equivalent PyTorch operation
