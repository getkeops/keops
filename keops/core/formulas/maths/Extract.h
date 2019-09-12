#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/ExtractT.h"
#include "core/pre_headers.h"

namespace keops {

template< class F, int START, int DIM_ >
struct ExtractT;

//////////////////////////////////////////////////////////////
////     VECTOR EXTRACTION : Extract<F,START,DIM>         ////
//////////////////////////////////////////////////////////////

template< class F, int START, int DIM_ >
struct Extract : UnaryOp< Extract, F, START, DIM_ > {
  static const int DIM = DIM_;

  static_assert(F::DIM >= START + DIM, "Index out of bound in Extract");
  static_assert(START >= 0, "Index out of bound in Extract");

  static void PrintIdString(::std::stringstream &str) {
    str << "Extract";
  }

  static HOST_DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outF) {
    for (int k = 0; k < DIM; k++)
      out[k] = outF[START + k];
  }

  template< class V, class GRADIN >
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template< class V, class GRADIN >
  using DiffT = DiffTF< V, ExtractT< GRADIN, START, F::DIM > >;
};

#define Extract(p,k,n) KeopsNS<Extract<decltype(InvKeopsNS(p)),k,n>>()

}
