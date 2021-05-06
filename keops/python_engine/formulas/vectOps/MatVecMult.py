from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.vectOps.TensorProd import TensorProd


# /////////////////////////////////////////////////////////////////////////
# ////     Matrix-vector product      A x b                           ////
# /////////////////////////////////////////////////////////////////////////


class MatVecMult(Operation):
    string_id = "MatVecMult"

    def __init__(self, A, B):
        # A is vector of size n*p, interpreted as matrix, B is vector of size p, interpreted as column vector
        # output is vector of size n
        if A.dim % B.dim != 0:
            raise ValueError("Dimensions of A and B are not compatible for matrix-vector product")
        super().__init__(A, B)
        self.dim = A.dim // B.dim

    def Op(self, out, table, inB, inA):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"""
                    #if C_CONTIGUOUS     // row major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                            #pragma unroll
                            for (int k = 0; k < {inB.dim}; k++, q++)
                                {out.id}[i] += {inA.id}[q] * {inB.id}[k];
                        }}
                    #else               // column major
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                        	#pragma unroll
                            for (int k = 0; k < {inB.dim}; k++)
                                {out.id}[i] += {inA.id}[{self.dim} * k + i] * {inB.id}[k];
                        }}
                    #endif
                """

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.vectOps.VecMatMult import VecMatMult

        A = self.children[0]
        B = self.children[1]
        return A.Grad(v, TensorProd(gradin, B)) + B.Grad(v, VecMatMult(gradin, A))