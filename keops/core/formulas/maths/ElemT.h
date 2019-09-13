#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Elem.h"
#include "core/pre_headers.h"

namespace keops {

template< class F, int M >
struct Elem;

//////////////////////////////////////////////////////////////
////     ELEMENT "INJECTION" : ElemT<F,N,M>               ////
//////////////////////////////////////////////////////////////

template< class F, int N, int M >
struct ElemT : UnaryOp< ElemT, F, N, M > {
  static const int DIM = N;
  static_assert(F::DIM == 1, "Input of ElemT should be a scalar");

  static void PrintIdString(::std::stringstream &str) {
    str << "ElemT";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
    for (int k = 0; k < DIM; k++)
      out[k] = 0.0;
    out[M] = *outF;
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT< V, GRADIN >;

  template< class V, class GRADIN >
  using DiffT = DiffTF< V, Elem< GRADIN, M > >;
};

#define ElemT(p,k) KeopsNS<ElemT<decltype(InvKeopsNS(p)),k>>()

}