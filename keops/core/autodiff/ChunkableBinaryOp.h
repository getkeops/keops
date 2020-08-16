#pragma once

#include <sstream>
#include "core/autodiff/BinaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class,class> class OP, class F, class G >
struct ChunkableBinaryOp : BinaryOp<OP, F, G> {
	
  template < int DIMCHK >
  using CHUNKED_VERSION = OP < typename F::template CHUNKED_VERSION<DIMCHK>, typename G::template CHUNKED_VERSION<DIMCHK> >;

  template < int CAT >
  using CHUNKED_VARS = MergePacks < typename F::template CHUNKED_VARS<CAT>, typename G::template CHUNKED_VARS<CAT> >;

  template < int CAT >
  using NOTCHUNKED_VARS = univpack<>;

  static const bool USE_CHUNK = ENABLECHUNK && F::IS_CHUNKABLE && G::IS_CHUNKABLE && F::DIM>100;

  template < int DIMCHK >
  using CHUNKED_FORMULA = CondType < univpack<univpack<CHUNKED_VERSION<DIMCHK>,pack<F::DIM>>>, univpack<>, USE_CHUNK >;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = ConcatPacks < ConcatPacks < typename F::template CHUNKED_FORMULAS<DIMCHK>, 
							typename G::template CHUNKED_FORMULAS<DIMCHK> >, CHUNKED_FORMULA<DIMCHK> >;

  static const int NUM_CHUNKED_FORMULAS = F::NUM_CHUNKED_FORMULAS + G::NUM_CHUNKED_FORMULAS + USE_CHUNK;

  template < int IND >
  using POST_CHUNK_FORMULA = CondType < Var < IND, 1, 3 >, OP<typename F::template POST_CHUNK_FORMULA<IND>, 
							typename G::template POST_CHUNK_FORMULA<IND>>, USE_CHUNK >;


};
  
}
