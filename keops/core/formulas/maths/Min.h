#pragma once

#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/ArgMin.h"
#include "core/formulas/maths/OneHot.h"

#include "core/pre_headers.h"
namespace keops {

//////////////////////////////////////////////////////////////
////                 MIN : Min< F >                       ////
//////////////////////////////////////////////////////////////

template<class F>
struct Min : UnaryOp<Min, F> {

  static_assert(F::DIM >= 1, "Min operation is only valid when dimension is non zero.");
  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) {
    str << "Min";
  }

  static DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
#if USE_HALF && GPU_ON
#elif USE_HALF
// this should never be used...
#else
    *out = outF[0];
#pragma unroll
    for (int k = 1; k < F::DIM; k++)
      if (outF[k] < *out)
		  *out = outF[k];
#endif
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Scal<GRADIN,OneHot<ArgMin<F>,F::DIM>>>;

};

#define Min(p) KeopsNS<Min<decltype(InvKeopsNS(p))>>()

}
