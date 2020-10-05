#pragma once

#include <assert.h>

#include "core/autodiff/ChunkableUnaryOp.h"
#include "core/formulas/maths/SumT.h"
#include "core/utils/TypesUtils.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                 SUM : Sum< F >                       ////
//////////////////////////////////////////////////////////////

template<class F, int D>
struct SumT;

template<class F>
struct Sum : ChunkableUnaryOp<Sum, F> {

  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) { str << "Sum"; }

  template < typename TYPE >
  struct Operation_Scalar {
  	DEVICE INLINE void operator() (TYPE& out, TYPE& outF) {
      	  out += outF;
	}
  };
  
  template < typename TYPE >
  static DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
    *out = cast_to<TYPE>(0.0f);
    VectApply < Operation_Scalar<TYPE>, DIM, F::DIM > (out, outF);
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, SumT<GRADIN, F::DIM>>;

  template < typename TYPE >
  static DEVICE INLINE void initacc_chunk(TYPE *acc) {
	*acc = 0.0f;
  }

  template < typename TYPE >
  static DEVICE INLINE void acc_chunk(TYPE *acc, TYPE *out) {
	*acc += *out;
  }

};

#define Sum(p) KeopsNS<Sum<decltype(InvKeopsNS(p))>>()

}
