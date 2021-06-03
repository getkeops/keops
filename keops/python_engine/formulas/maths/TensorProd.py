from keops.python_engine.formulas.maths.Operation import Operation

####################################
######    Tensor product       #####
####################################

class TensorProd(Operation):
    string_id = "TensorProd"

    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim * arg1.dim

    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"""
                    #if C_CONTIGUOUS     // row major
                        int q = 0;
                        #pragma unroll
                        for (int k = 0; k < {arg0.dim}; k++) 
                        {{
                            #pragma unroll
                            for (int l = 0; l < {arg1.dim}; l++, q++)
                                {out.id}[q] = {arg0.id}[k] * {arg1.id}[l];
                        }}
                    #else               // column major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {arg0.dim}; i++) 
                        {{
                            #pragma unroll
                            for (int j = 0; j < {arg1.dim}; j++, q++)
                                {out.id}[{arg0.dim} * j + i] = {arg0.id}[i] * {arg1.id}[j];
                        }}
                    #endif
                """

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas import MatVecMult, VecMatMult
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v, MatVecMult(gradin, g)) + g.Grad(v, VecMatMult(f, gradin))