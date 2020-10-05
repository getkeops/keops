#pragma once

#include <sstream>
#include "core/autodiff/UnaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class,int...> class OP, class F, int... NS >
struct ChunkableUnaryOp : UnaryOp<OP, F, NS...> {
	
    // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
    /*
    template < int DIMCHK >
    using CHUNKED_VERSION = OP<typename F::template CHUNKED_VERSION<DIMCHK>,NS...>;
    */
    // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack : 
    template < int DIMCHK, int SIZE_NS >
    struct CHUNKED_VERSION_Impl {
      using type = OP<typename F::template CHUNKED_VERSION<DIMCHK>,NS...>;
    };
    
    template < int DIMCHK >
    struct CHUNKED_VERSION_Impl < DIMCHK, 0 > {
      using type = OP<typename F::template CHUNKED_VERSION<DIMCHK> >;
    };
    
    template < int DIMCHK >
    using CHUNKED_VERSION = typename CHUNKED_VERSION_Impl<DIMCHK,sizeof...(NS)>::type;
    
  template < int CAT >
  using CHUNKED_VARS = typename F::template CHUNKED_VARS<CAT>;

  template < int CAT >
  using NOTCHUNKED_VARS = univpack<>;

  static const bool USE_CHUNK = ENABLECHUNK && F::IS_CHUNKABLE 
				&& (F::DIM>DIM_TRESHOLD_CHUNK || F::DIM==SPECDIM_USE_CHUNK1 || F::DIM==SPECDIM_USE_CHUNK2 
				|| F::DIM==SPECDIM_USE_CHUNK3 || F::DIM==SPECDIM_USE_CHUNK4);

  template < int DIMCHK >
  using CHUNKED_FORMULA = CondType < univpack<univpack<CHUNKED_VERSION<DIMCHK>,pack<F::DIM>>>, univpack<>, USE_CHUNK >;

  template < int DIMCHK >
  using CHUNKED_FORMULAS = ConcatPacks < typename F::template CHUNKED_FORMULAS<DIMCHK>, CHUNKED_FORMULA<DIMCHK> >;

  static const int NUM_CHUNKED_FORMULAS = F::NUM_CHUNKED_FORMULAS + USE_CHUNK;


  // NB. The following commented code should be ok but it dos not compile with Cuda 11 as of 2020 aug 13th...
  /*
  template < int IND >
  using POST_CHUNK_FORMULA = CondType < Var < IND, 1, 3 >, OP<typename F::template POST_CHUNK_FORMULA<IND>, NS...>, USE_CHUNK >;
  */
  // ... so we use an additional "_Impl" structure to specialize in case of empty PARAMS pack : 
  template < int IND, int SIZE_NS >
  struct POST_CHUNK_FORMULA_Impl {
	  using type = CondType < Var < IND, 1, 3 >, OP<typename F::template POST_CHUNK_FORMULA<IND>, NS...>, USE_CHUNK >;
  };
  
  template < int IND >
  struct POST_CHUNK_FORMULA_Impl < IND, 0 > {
	  using type = CondType < Var < IND, 1, 3 >, OP<typename F::template POST_CHUNK_FORMULA<IND>>, USE_CHUNK >;
  };
  
  template < int IND >
  using POST_CHUNK_FORMULA = typename POST_CHUNK_FORMULA_Impl<IND,sizeof...(NS)>::type;
  
};
  
}
