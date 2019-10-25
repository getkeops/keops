#pragma once

#include <sstream>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Step.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             RELU : ReLU< F >                         ////
//////////////////////////////////////////////////////////////

template<class F>
struct ReLU : UnaryOp<ReLU, F> {
  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "ReLU";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#if USE_HALF && GPU_ON
#pragma unroll
    for (int k = 0; k < DIM; k++)
      if hlt(outF[k],0)
        out[k] = 0.0;
      else
        out[k] = outF[k];
#elif USE_HALF
#pragma unroll
    for (int k = 0; k < DIM; k++)
      if (outF[k] < (half)0)
        out[k] = 0.0;
      else
        out[k] = outF[k];
#else
#pragma unroll
    for (int k = 0; k < DIM; k++)
      if (outF[k] < 0)
        out[k] = 0.0;
      else
        out[k] = outF[k];
#endif
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Step<F>, GRADIN>>;
};

#define ReLU(f) KeopsNS<ReLU<decltype(InvKeopsNS(f))>>()

}
