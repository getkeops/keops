#pragma once

#include <sstream>

#include "core/autodiff/UnaryOp.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////               MINUS OPERATOR : Minus< F >            ////
//////////////////////////////////////////////////////////////

template<class F>
struct Minus : UnaryOp<Minus, F> {

  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "Minus";
  }

  static DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = -outF[k];
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Minus<GRADIN>>;

};

template < class F >
KeopsNS<Minus<F>> operator-(KeopsNS<F> f) {
  return KeopsNS<Minus<F>>();
}
#define Minus(f) KeopsNS<Minus<decltype(InvKeopsNS(f))>>()


}
