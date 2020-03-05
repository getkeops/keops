#pragma once

#include <assert.h>

#include "core/formulas/constants/Zero.h"
#include "core/autodiff/UnaryOp.h"

#include "core/pre_headers.h"
namespace keops {

//////////////////////////////////////////////////////////////
////                 ARGMin : ArgMin< F >                 ////
//////////////////////////////////////////////////////////////

template<class F>
struct ArgMin : UnaryOp<ArgMin, F> {

  static_assert(F::DIM >= 1, "[KeOps] ArgMin operation is only possible when dimension is non zero.");

  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) { str << "ArgMin"; }

  template < typename TYPE >
  static DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
    *out = 0.0f;
    TYPE tmp = outF[0];
    #pragma unroll
    for (int k = 1; k < F::DIM; k++)
      if (outF[k] < tmp) {
      	tmp = outF[k];
	*out = k;  
      }  
  }

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

  template<class V, class GRADIN>
  using DiffT = Zero<V::DIM>;

};

#define ArgMin(p) KeopsNS<ArgMin<decltype(InvKeopsNS(p))>>()

}
