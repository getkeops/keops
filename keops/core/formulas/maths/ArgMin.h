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

#if __TYPE__==float
  static_assert(F::DIM <= 16777216, "[KeOps] Dimension is too large for storing indices as single precision floats.");
#endif

  static_assert(F::DIM >= 1, "ArgMin operation is only valid when dimension is non zero.");
  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) {
    str << "ArgMin";
  }

  static DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
    *out = 0.0;
	__TYPE__ tmp = outF[0];
#pragma unroll
    for (int k = 0; k < F::DIM; k++)
      if (outF[k] < tmp) {
      	tmp = outF[k];
		*out = k;
      }
  }

  template<class V, class GRADIN>
  using DiffT = Zero<V::DIM>;

};

#define ArgMin(p) KeopsNS<ArgMin<decltype(InvKeopsNS(p))>>()

}
