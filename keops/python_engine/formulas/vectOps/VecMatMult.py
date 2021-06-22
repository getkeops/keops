from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.vectOps.TensorProd import TensorProd
from keops.python_engine.utils.code_gen_utils import c_variable, c_for_loop, c_zero_float


# /////////////////////////////////////////////////////////////////////////
# ////     Vector-matrix product           b x A                       ////
# /////////////////////////////////////////////////////////////////////////

class VecMatMult(Operation):
    string_id = "VecMatMult"

    def __init__(self, B, A):
        # A is vector of size n*p, interpreted as matrix, B is vector of size n, interpreted as row vector
        # output is vector of size p
        if A.dim % B.dim != 0:
            raise ValueError("Dimensions of A and B are not compatible for vector-matrix product")
        super().__init__(B, A)
        self.dim = A.dim // B.dim

    def Op(self, out, table, inB, inA):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        q = c_variable("int")
        loop, i = c_for_loop(0, self.dim, 1, pragma_unroll=True)
        inner_loop, k = c_for_loop(0, inB.dim, 1, pragma_unroll=True)
        return f"""
                    #if C_CONTIGUOUS     // row major
                        {q.declare_assign(0)}
                        {loop( out[i].assign(c_zero_float) + inner_loop( out[i].add_assign(inA[k*self.dim+i]*inB[k])) )}
                    #else               // column major
                        {q.declare_assign(0)}
                        {loop( out[i].assign(c_zero_float) + inner_loop( out[i].add_assign(inA[q]*inB[k]) + q.add_assign(1) ) )}
                    #endif
                """

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.maths.MatVecMult import MatVecMult
        B = self.children[0]
        A = self.children[1]
        return A.DiffT(v, TensorProd(B, gradin)) + B.DiffT(v, MatVecMult(A, gradin))