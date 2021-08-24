#pragma once

#include <sstream>
#include "core/autodiff/UnaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class,int...> class OP, class F, int... NS >
struct VectorizedScalarUnaryOp : UnaryOp<OP, F, NS...> {
	
    static const int DIM = F::DIM;

    static const bool IS_CHUNKABLE = F::IS_CHUNKABLE;
  
  
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
    using NOTCHUNKED_VARS = typename F::template NOTCHUNKED_VARS<CAT>;

    template < typename TYPE >
    static DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
	using OpScal = typename OP<F,NS...>::template Operation_Scalar<TYPE>;
  	VectApply < OpScal, DIM, DIM > (out, outF);
    }

};
  
}
