#pragma once

#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"
#include "core/formulas/maths/maths.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Step.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             RELU : ReLU< F >                         ////
//////////////////////////////////////////////////////////////

template<class F>
struct ReLU : UnaryOp<ReLU, F> {
  static const int DIM = F::DIM;

  static void PrintIdString(std::stringstream &str) {
    str << "ReLU";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      if (outF[k] < 0)
        out[k] = 0.0;
      else
        out[k] = outF[k];
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Step<F>, GRADIN>>;
};

}