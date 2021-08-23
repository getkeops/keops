from keops.formulas.Operation import Operation
from keops.formulas.maths.ArgMax import ArgMax
from keops.formulas.maths.OneHot import OneHot
from keops.utils.code_gen_utils import c_for_loop, c_if, value


############################
######    Max       #####
############################

class Max(Operation):
    string_id = "Max"

    def __init__(self, f):
        super().__init__(f)
        if f.dim < 1:
            raise ValueError("[KeOps] Max operation is only possible when dimension is non zero.")
        self.dim = 1
        self.argdim = f.dim

    def Op(self, out, table, arg):
        loop, k = c_for_loop(1, arg.dim, 1, pragma_unroll=True)
        string = value(out).assign(arg[0])
        string += loop(c_if(arg[k] > value(out), value(out).assign(arg[k])))
        return string

    def DiffT(self, v, gradin):
        return f.DiffT(v, OneHot(ArgMax(f), self.argdim) * gradin)  # TODO : Fix this
        
    
    
    # parameters for testing the operation (optional)
    enable_test = True          # enable testing for this operation
    nargs = 1                   # number of arguments
    test_argdims = [5]          # dimensions of arguments for testing
    torch_op = "lambda x : torch.max(x, dim=-1, keepdim=True)"




# TODO : transcript half2 implementation below

"""
#if USE_HALF && GPU_ON
  static DEVICE INLINE void Operation(half2 *out, half2 *outF) {
    *out = outF[0];
    #pragma unroll
    for (int k = 1; k < F::DIM; k++) {
      // we have to work element-wise...
      __half2 cond = __hlt2(*out,outF[k]);                 // cond = (out < outF[k]) (element-wise)
      __half2 negcond = __float2half2_rn(1.0f)-cond;       // negcond = 1-cond
      *out = cond * outF[k] + negcond * *out;              // out  = cond * outF[k] + (1-cond) * out
    }
  }
#endif
"""