#pragma once

#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"
#include "core/formulas/maths/maths.h"
#include "core/formulas/maths/Mult.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             EXPONENTIAL : Exp< F >                   ////
//////////////////////////////////////////////////////////////

template<class F>
struct Exp : UnaryOp<Exp, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(std::stringstream &str) {
    str << "Exp";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
#ifdef __NVCC__
#if USE_DOUBLE
      out[k] = exp(outF[k]);
#else
      out[k] = expf(outF[k]);
#endif
#else
        out[k] = out[k] = std::exp(outF[k]);
#endif
  }

  // [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Exp<F>, GRADIN>>;

};

}
