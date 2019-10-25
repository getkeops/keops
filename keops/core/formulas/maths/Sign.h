#pragma once

#include <sstream>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/constants/Zero.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             SIGN : Sign< F >                         ////
//////////////////////////////////////////////////////////////

template<class F>
struct Sign : UnaryOp<Sign, F> {
  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "Sign";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#if USE_HALF && GPU_ON
#pragma unroll
    for (int k = 0; k < DIM; k++)
      if hgt(outF[k],0)
        out[k] = 1.0;
      else if hlt(outF[k],0)
        out[k] = -1.0;
      else
        out[k] = 0.0;
#elif USE_HALF
    for (int k = 0; k < DIM; k++)
      if (outF[k] > (half)0)
        out[k] = (half)1.0;
      else if (outF[k] < (half)0)
        out[k] = (half)-1.0;
      else
        out[k] = (half)0.0;
#else
#pragma unroll
    for (int k = 0; k < DIM; k++)
      if (outF[k] > 0)
        out[k] = 1.0;
      else if (outF[k] < 0)
        out[k] = -1.0;
      else
        out[k] = 0.0;
#endif
  }

  template<class V, class GRADIN>
  using DiffT = Zero<V::DIM>;
};

#define Sign(f) KeopsNS<Sign<decltype(InvKeopsNS(f))>>()

}
