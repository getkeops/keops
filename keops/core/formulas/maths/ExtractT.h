#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Extract.h"
#include "core/pre_headers.h"

namespace keops {

template< class F, int START, int DIM_ >
struct Extract;

//////////////////////////////////////////////////////////////
////     VECTOR "INJECTION" : ExtractT<F,START,DIM>       ////
//////////////////////////////////////////////////////////////

template < class F, int START, int DIM_ >
struct ExtractT : UnaryOp<ExtractT,F,START,DIM_> {
  static const int DIM = DIM_;

  static_assert(START+F::DIM<=DIM,"Index out of bound in ExtractT");
  static_assert(START>=0,"Index out of bound in ExtractT");

  static void PrintIdString(::std::stringstream& str) {
    str << "ExtractT";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
    for(int k=0; k<START; k++)
      out[k] = 0.0;
    for(int k=0; k<F::DIM; k++)
      out[START+k] = outF[k];
    for(int k=START+F::DIM; k<DIM; k++)
      out[k] = 0.0;
  }

  template < class V, class GRADIN >
  using DiffTF = typename F::template DiffT<V,GRADIN>;

  template < class V, class GRADIN >
  using DiffT = DiffTF<V,Extract<GRADIN,START,F::DIM>>;
};

#define ExtractT(p,k,n) KeopsNS<ExtractT<decltype(InvKeopsNS(p)),k,n>>()

}
