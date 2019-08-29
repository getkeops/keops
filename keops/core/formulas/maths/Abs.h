#pragma once

#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"
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
    for (int k = 0; k < DIM; k++)
      if (outF[k] < 0)
        out[k] = -outF[k];
      else
        out[k] = outF[k];
  }

  // [\partial_V abs(F)].gradin = sign(F) * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Sign<F>, GRADIN>>;

};


}