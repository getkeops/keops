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
#pragma unroll
    for (int k = 0; k < DIM; k++) 
#if USE_HALF && GPU_ON
      out[k] =  __hgt2(outF[k],__float2half2_rn(0.0f)) - __hlt2(outF[k],__float2half2_rn(0.0f));  // (outF[k]>0) - (outF[k]<0) (element-wise)
#elif USE_HALF
{}
// this should never be used...
#else
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
