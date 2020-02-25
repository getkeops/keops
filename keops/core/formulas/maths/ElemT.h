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

  static void PrintIdString(::std::stringstream &str) { str << "ElemT"; }

  template < typename TYPE >
  static HOST_DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
    VectAssign<M>(out, 0.0f);
    out[M] = *outF;
    VectAssign<N-M-1>(out+M+1, 0.0f);
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT< V, GRADIN >;

  template< class V, class GRADIN >
  using DiffT = DiffTF< V, Elem< GRADIN, M > >;
};

#define ElemT(p,k) KeopsNS<ElemT<decltype(InvKeopsNS(p)),k>>()

}
