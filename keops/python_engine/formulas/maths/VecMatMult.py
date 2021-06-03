from keops.python_engine.formulas.maths.Operation import Operation
from keops.python_engine.formulas.maths.TensorProd import TensorProd


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
        return f"""
                    #if C_CONTIGUOUS     // row major
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                        	#pragma unroll
                            for (int k = 0; k < {inB.dim}; k++)
                                {out.id}[i] += {inA.id}[{self.dim} * k + i] * {inB.id}[k];
                        }}
                    #else               // column major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                            #pragma unroll
                            for (int k = 0; k < {inB.dim}; k++, q++)
                                {out.id}[i] += {inA.id}[q] * {inB.id}[k];
                        }}
                    #endif
                """

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.maths.MatVecMult import MatVecMult
        B = self.children[0]
        A = self.children[1]
        return A.Grad(v, TensorProd(B, gradin)) + B.Grad(v, MatVecMult(A, gradin))