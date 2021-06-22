#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/ElemT.h"
#include "core/pre_headers.h"

namespace keops {

template< class F, int N, int M >
struct ElemT;

//////////////////////////////////////////////////////////////
////     ELEMENT EXTRACTION : Elem<F,M>                   ////
//////////////////////////////////////////////////////////////

template< class F, int M >
struct Elem : UnaryOp< Elem, F, M > {

  static const int DIM = 1;
  static_assert(F::DIM > M, "Index out of bound in Elem");

  static void PrintIdString(::std::stringstream &str) { str << "Elem"; }

  template < typename TYPE > 
  static HOST_DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
    *out = outF[M];
  }

  template< class V, class GRADIN >
  using DiffTF = typename F::template DiffT< V, GRADIN >;

  template< class V, class GRADIN >
  using DiffT = DiffTF< V, ElemT< GRADIN, F::DIM, M > >;
};

#define Elem(p,k) KeopsNS<Elem<decltype(InvKeopsNS(p)),k>>()

}
