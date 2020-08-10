#pragma once

#include <assert.h>

#include "core/autodiff/UnaryOp.h"
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
struct Sum : UnaryOp<Sum, F> {

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
    VectApply < Operation_Scalar<TYPE>, F::DIM > (*out, outF);
  }

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, SumT<GRADIN, F::DIM>>;

  template < int DIMCHK >
  using CHUNKED_VERSION = Sum < typename F::template CHUNKED_VERSION<DIMCHK> >;

  template < int CAT >
  using CHUNKED_VARS = typename F::template CHUNKED_VARS<CAT>;

  template < int CAT >
  using NOTCHUNKED_VARS = univpack<>;

  static const bool IS_CHUNKABLE = false;

  static const bool USE_CHUNK = ENABLECHUNK && F::IS_CHUNKABLE && F::DIM>100;

  template < int DIMCHK >
  using CHUNKED_FORMULA = CondType < univpack<univpack<CHUNKED_VERSION<DIMCHK>,pack<F::DIM>>>, univpack<>, USE_CHUNK >;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = ConcatPacks < typename F::template CHUNKED_FORMULAS<DIMCHK>, CHUNKED_FORMULA<DIMCHK> >;

  static const int NUM_CHUNKED_FORMULAS = F::NUM_CHUNKED_FORMULAS + USE_CHUNK;

  template < int IND >
  using POST_CHUNK_FORMULA = CondType < Var < IND, 1, 3 >, Sum<typename F::template POST_CHUNK_FORMULA<IND>>, USE_CHUNK >;

  template < typename TYPE >
  static DEVICE INLINE void initacc_chunk(TYPE *acc) {
	*acc = 0.0f;
  }

  template < typename TYPE >
  static DEVICE INLINE void acc_chunk(TYPE *acc, TYPE *out) {
	*acc += *out;
  }

  //template < int IND >
  //using POST_CHUNK_FORMULA = CondType < USE_CHUNK, Var<IND,DIM,4>, Sum<F::POST_CHUNK_FORMULA<IND>> >;

};

#define Sum(p) KeopsNS<Sum<decltype(InvKeopsNS(p))>>()

}
