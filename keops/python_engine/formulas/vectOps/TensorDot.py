from keops.python_engine.formulas.Operation import Operation
import numpy as np


####################################
######  Tensor Dot Product     #####
####################################

class TensorDot(Operation):
    string_id = "TensorDot"

    def __init__(self, fa, fb, dimsfa, dimsfb, contfa, contfb, permute=None):

        assert (dimsfb[contfb] == dimsfa[contfa])

        assert (fa.dim == dimsfa.prod())
        assert (fb.dim == dimsfb.prod())

        super().__init__(fa, fb)

        self.contdims = dimsfa[contfa]
        self.keepdims = np.concatenate((np.delete(dimsfa, contfa), np.delete(dimsfb, contfb)))

        self.dim = fa.dim * fb.dim
        self.dim /= self.contdims.prod() ** 2 if len(contfa) else 1

        if permute is None:
            permute = np.arange(self.dim)

        self.permute = permute

        self.list_strides_dimsfa = np.cumprod_array(dimsfa)

    @staticmethod
    def cumprod_array(x):
        if len(x) == 0:
            return x
        else:
            return np.concatenate((np.cumprod(x[1:][::-1])[::-1], [1]))

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
                        
                    #endif
                """

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas import MatVecMult, VecMatMult
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v, MatVecMult(gradin, g)) + g.Grad(v, VecMatMult(f, gradin))
