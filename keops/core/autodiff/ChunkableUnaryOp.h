#pragma once

#include <sstream>
#include "core/autodiff/UnaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class,int...> class OP, class F, int... NS >
struct ChunkableUnaryOp : UnaryOp<OP, F, NS...> {
	
  template < int DIMCHK >
  using CHUNKED_VERSION = OP < typename F::template CHUNKED_VERSION<DIMCHK>, NS... >;

  template < int CAT >
  using CHUNKED_VARS = typename F::template CHUNKED_VARS<CAT>;

  template < int CAT >
  using NOTCHUNKED_VARS = univpack<>;

  static const bool USE_CHUNK = ENABLECHUNK && F::IS_CHUNKABLE && F::DIM>100;

  template < int DIMCHK >
  using CHUNKED_FORMULA = CondType < univpack<univpack<CHUNKED_VERSION<DIMCHK>,pack<F::DIM>>>, univpack<>, USE_CHUNK >;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = ConcatPacks < typename F::template CHUNKED_FORMULAS<DIMCHK>, CHUNKED_FORMULA<DIMCHK> >;

  static const int NUM_CHUNKED_FORMULAS = F::NUM_CHUNKED_FORMULAS + USE_CHUNK;

  template < int IND >
  using POST_CHUNK_FORMULA = CondType < Var < IND, 1, 3 >, OP<typename F::template POST_CHUNK_FORMULA<IND>, NS...>, USE_CHUNK >;


};
  
}
