#pragma once

#include <assert.h>

#include "core/utils/TypesUtils.h"
#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Sum.h"

#include "core/pre_headers.h"
namespace keops {

//////////////////////////////////////////////////////////////
////        Transpose of Sum : SumT< F >                   ////
//////////////////////////////////////////////////////////////
template<class F>
struct Sum;

template<class F, int D>
struct SumT : UnaryOp<SumT, F, D> {

  static_assert(F::DIM == 1, "Dimension of input must be 1 for SumT");

  static const int DIM = D;

  static void PrintIdString(::std::stringstream& str) { str << "SumT"; }

  template < typename TYPE >
  static DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
    VectAssign<DIM>(out, *outF);
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Sum<GRADIN>>;

};

#define SumT(p,d) KeopsNS<SumT<decltype(InvKeopsNS(p)),d>>()

}
