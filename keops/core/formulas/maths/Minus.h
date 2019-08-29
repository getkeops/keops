#pragma once

#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"

namespace keops {

//////////////////////////////////////////////////////////////
////               MINUS OPERATOR : Minus< F >            ////
//////////////////////////////////////////////////////////////

template<class F>
struct Minus : UnaryOp<Minus, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(std::stringstream &str) {
    str << "Minus";
  }

  static HOST_DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = -outF[k];
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Minus<GRADIN>>;

};

}