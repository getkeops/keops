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
struct Sum_Impl;

template<class F, int D>
struct SumT_Impl : UnaryOp<SumT_Impl, F, D> {

  static_assert(F::DIM == 1, "Dimension of input must be 1 for SumT");

  static const int DIM = D;

  static void PrintIdString(::std::stringstream& str) { str << "SumT"; }

  template < typename TYPE >
  static DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
    VectAssign<DIM>(out, *outF);
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Sum_Impl<GRADIN>>;

};

template < class F, int D >
struct SumT_Alias {
	using type = SumT_Impl<F,D>;
};

template < class F >
struct SumT_Alias<F,1> {
	static_assert(F::DIM == 1, "Dimension of input must be 1 for SumT");
	using type = F;
};

template < class F, int D >
using SumT = typename SumT_Alias<F,D>::type;

#define SumT(p,d) KeopsNS<SumT<decltype(InvKeopsNS(p)),d>>()

}
