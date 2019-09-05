#pragma once

#include <sstream>

#include "core/autodiff.h"
#include "core/formulas/maths/Sin.h"
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Mult.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                 COSINE :  Cos< F >                   ////
//////////////////////////////////////////////////////////////


template<class F>
struct Sin;


template<class F>
struct Cos : UnaryOp<Cos, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(std::stringstream &str) {
    str << "Cos";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = cos(outF[k]);
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Minus<Mult<Sin<F>, GRADIN>>>;

};

#define Cos(f) KeopsNS<Cos<decltype(InvKeopsNS(f))>>()

}
