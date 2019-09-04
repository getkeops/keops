#pragma once

#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"
#include "core/formulas/maths/maths.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Cos.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                  SINE :  Sin< F >                    ////
//////////////////////////////////////////////////////////////

template<class F>
struct Cos;

template<class F>
struct Sin : UnaryOp<Sin, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(std::stringstream &str) {
    str << "Sin";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = sin(outF[k]);
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Cos<F>, GRADIN>>;

};

}
