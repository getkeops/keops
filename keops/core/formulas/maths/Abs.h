#pragma once

#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"
#include "core/formulas/maths/maths.h"
#include "core/formulas/maths/Sign.h"
#include "core/formulas/maths/Mult.h"

namespace keops {

//////////////////////////////////////////////////////////////
////           ABSOLUTE VALUE : Abs< F >                  ////
//////////////////////////////////////////////////////////////

template<class F>
struct Abs : UnaryOp<Abs, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(std::stringstream &str) {
    str << "Abs";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++){
#ifdef __NVCC__
#if USE_DOUBLE
        out[k] = fabs(outF[k]);
#else
        out[k] = fabsf(outF[k]);
#endif
#else
      out[k] =  std::abs(outF[k]);
#endif
    }
  }

  // [\partial_V abs(F)].gradin = sign(F) * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Sign<F>, GRADIN>>;

};


}