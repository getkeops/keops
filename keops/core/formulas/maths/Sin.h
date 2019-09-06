#pragma once

#include <sstream>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Cos.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                  SINE :  Sin< F >                    ////
//////////////////////////////////////////////////////////////

template<class F>
struct Cos;

template<class F>
struct Sin : UnaryOp<Sin, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "Sin";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++) {
#if USE_DOUBLE
      out[k] = sin(outF[k]);
#else
      out[k] = sinf(outF[k]);
#endif
    }
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Cos<F>, GRADIN>>;

};

#define Sin(f) KeopsNS<Sin<decltype(InvKeopsNS(f))>>()

}
