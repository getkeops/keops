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
#elif USE_HALF
#if GPU_ON
      // There is a exp operation for half2 type
      //out[k] = h2exp(outF[k]);
      // but doing experiments on RTX 2080 Ti card, it appears to be very slow, 
      // so we use expf function twice instead :
      float a = expf(__low2float(outF[k]));
      float b = expf(__high2float(outF[k]));
      out[k] = __floats2half2_rn(a,b);
#else
// we should never use this...
#endif
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
