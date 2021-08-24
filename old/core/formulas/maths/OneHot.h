#pragma once

#include <sstream>
#include <assert.h>

#include "core/utils/TypesUtils.h"
#include "core/formulas/constants/Zero.h"
#include "core/autodiff/UnaryOp.h"
#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////       ONE-HOT REPRESENTATION : OneHot<F,DIM>         ////
//////////////////////////////////////////////////////////////

template< class F, int DIM_ >
struct OneHot : UnaryOp< OneHot, F, DIM_ > {
  static const int DIM = DIM_;

  static_assert(F::DIM == 1, "One-hot representation is only supported for scalar formulas.");
  static_assert(DIM_ >= 1, "A one-hot vector should have length >= 1.");

  static void PrintIdString(::std::stringstream &str) { str << "OneHot"; }

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
