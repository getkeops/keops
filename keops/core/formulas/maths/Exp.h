#pragma once

#include <sstream>
#include <cmath>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Mult.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             EXPONENTIAL : Exp< F >                   ////
//////////////////////////////////////////////////////////////

template<class F>
struct Exp : UnaryOp<Exp, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "Exp";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++) {
#if USE_DOUBLE
      out[k] = exp(outF[k]);
#else
      out[k] = expf(outF[k]);
#endif
    }
  }

  // [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Exp<F>, GRADIN>>;

};

#define Exp(f) KeopsNS<Exp<decltype(InvKeopsNS(f))>>()

}
