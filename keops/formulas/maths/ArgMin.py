from keops.formulas.Operation import Operation
from keops.formulas.variables.Zero import Zero
from keops.utils.code_gen_utils import c_zero_float, c_for_loop, c_if, value, c_variable


############################
######    ArgMin       #####
############################

class ArgMin(Operation):
    string_id = "ArgMin"

    def __init__(self, f):
        super().__init__(f)
        if f.dim < 1:
            raise ValueError("[KeOps] ArgMin operation is only possible when dimension is non zero.")
        self.dim = 1

    def Op(self, out, table, arg):
        tmp = c_variable(out.dtype)
        loop, k = c_for_loop(1, arg.dim, 1, pragma_unroll=True)
        string = value(out).assign(c_zero_float) + tmp.declare_assign(arg[0])
        string += loop(c_if(arg[k] < tmp, tmp.assign(arg[k]) + value(out).assign(k)))
        return string

    def DiffT(self, v, gradin):
        return Zero(v.dim)

    
    
    # parameters for testing the operation (optional)
    enable_test = True          # enable testing for this operation
    nargs = 1                   # number of arguments
    test_argdims = [5]          # dimensions of arguments for testing
    torch_op = "lambda x : torch.argmin(x, dim=-1, keepdim=True).type(x.dtype)"


# TODO : half2 implementation below
"""
#if USE_HALF && GPU_ON
  static DEVICE INLINE void Operation(half2 *out, half2 *outF) {
    *out = __float2half2_rn(0.0f);
    half2 tmp = outF[0];
    #pragma unroll
    for (int k = 1; k < F::DIM; k++) {
      // we have to work element-wise...
      __half2 cond = __hgt2(tmp,outF[k]);                  // cond = (tmp > outF[k]) (element-wise)
      __half2 negcond = __float2half2_rn(1.0f)-cond;       // negcond = 1-cond
      *out = cond * __float2half2_rn(k) + negcond * *out;  // out  = cond * k + (1-cond) * out 
      tmp = cond * outF[k] + negcond * tmp;                // tmp  = cond * outF[k] + (1-cond) * tmp
    }  
  }
#endif
"""