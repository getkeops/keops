#pragma once

#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/ArgMax.h"
#include "core/formulas/maths/OneHot.h"

#include "core/pre_headers.h"
namespace keops {

//////////////////////////////////////////////////////////////
////                 MAX : Max< F >                       ////
//////////////////////////////////////////////////////////////

template<class F>
struct Max : UnaryOp<Max, F> {

  static_assert(F::DIM >= 1, "Max operation is only valid when dimension is non zero.");
  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) {
    str << "Max";
  }

  static DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
    *out = outF[0];
#pragma unroll
    for (int k = 1; k < F::DIM; k++)
      if (outF[k] > *out)
		  *out = outF[k];
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Scal<GRADIN,OneHot<ArgMax<F>,F::DIM>>>;

};

#define Max(p) KeopsNS<Max<decltype(InvKeopsNS(p))>>()

}
