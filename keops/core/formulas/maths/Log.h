#pragma once

#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"
#include "core/formulas/maths/maths.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Inv.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             LOGARITHM : Log< F >                     ////
//////////////////////////////////////////////////////////////

template<class F>
struct Log : UnaryOp<Log, F> {
  static const int DIM = F::DIM;

  static void PrintIdString(std::stringstream &str) {
    str << "Log";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = log(outF[k]);
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Inv<F>, GRADIN>>;
};

}