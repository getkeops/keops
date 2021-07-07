from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.utils.code_gen_utils import (
    c_variable,
    c_for_loop,
    c_zero_float,
)

# /////////////////////////////////////////////////////////////////////////
# ////     Matrix-vector product      A x b                           ////
# /////////////////////////////////////////////////////////////////////////


class MatVecMult(Operation):
    string_id = "MatVecMult"

    def __init__(self, A, B):
        # A is vector of size n*p, interpreted as matrix, B is vector of size p, interpreted as column vector
        # output is vector of size n
        if A.dim % B.dim != 0:
            raise ValueError(
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
                        {loop( out[i].assign(c_zero_float) + inner_loop( out[i].add_assign(inA[q]*inB[k]) + q.add_assign(1) ) )}
                    #else               // column major
                        {loop( out[i].assign(c_zero_float) + inner_loop( out[i].add_assign(inA[k*self.dim+i]*inB[k]) ) )}
                    #endif
                """

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.vectOps.TensorProd import TensorProd
        from keops.python_engine.formulas.maths.VecMatMult import VecMatMult

        A, B = self.children
        return A.DiffT(v, TensorProd(gradin, B)) + B.DiffT(v, VecMatMult(gradin, A))
