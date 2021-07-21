from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.utils.code_gen_utils import c_array, VectCopy


# //////////////////////////////////////////////////////////////
# ////       ONE-HOT REPRESENTATION : OneHot<F,DIM>         ////
# //////////////////////////////////////////////////////////////


class OneHot(Operation):
    raise ValueError("One-hot is not yet implemented.")

    string_id = "OneHot"

    def __init__(self, f, dim):
        if f.dim != 1:
            raise ValueError(
                "One-hot representation is only supported for scalar formulas."
            )
        if dim < 1:
            raise ValueError("A one-hot vector should have length >= 1.")
        super().__init__(f)
        self.dim = dim
        self.params = (dim,)

    def Op(self, out, table, arg0):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        v = c_array(arg0.dtype, out.dim, f"({arg0.id}+{self.start})")
        return VectCopy(out, v)

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.maths.ExtractT import ExtractT

        f = self.children[0]
        return f.DiffT(v, ExtractT(gradin, self.start, f.dim))


"""

TO DO....


  template < typename TYPE >
  static HOST_DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
    VectAssign<DIM>(out, 0.0f);
    out[(int)(*outF+.5)] = 1.0;
  }

#if USE_HALF && GPU_ON
  static HOST_DEVICE INLINE void Operation(half2 *out, half2 *outF) {
    #pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = __heq2(h2rint(*outF),__float2half2_rn(k));
  }
#endif

  // There is no gradient to accumulate on V, whatever V.
  template < class V, class GRADIN >
  using DiffT = Zero<V::DIM>;
};

#define OneHot(f,n) KeopsNS<OneHot<decltype(InvKeopsNS(f)),n>>()

}
"""
