#pragma once

#include <assert.h>

#include "core/formulas/constants/Zero.h"
#include "core/autodiff/UnaryOp.h"

#include "core/pre_headers.h"
namespace keops {

//////////////////////////////////////////////////////////////
////                 ARGMIN : ArgMin< F >                 ////
//////////////////////////////////////////////////////////////

template<class F>
struct ArgMin : UnaryOp<ArgMin, F> {

  static_assert(F::DIM >= 1, "ArgMin operation is only valid when dimension is non zero.");
  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) {
    str << "ArgMin";
  }

  static DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
#if USE_HALF && GPU_ON
//#pragma unroll
#elif USE_HALF
// this should never be used...
#else
    *out = 0.0;
	__TYPE__ tmp = outF[0];
#pragma unroll
    for (int k = 1; k < F::DIM; k++)
      if (outF[k] < tmp) {
      	tmp = outF[k];
		*out = k;
      }
#endif

  }

  template<class V, class GRADIN>
  using DiffT = Zero<V::DIM>;

};

#define ArgMin(p) KeopsNS<ArgMin<decltype(InvKeopsNS(p))>>()

}
