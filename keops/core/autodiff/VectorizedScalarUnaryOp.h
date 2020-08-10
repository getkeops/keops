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

    template < int DIMCHK >
    using CHUNKED_VERSION = OP<typename F::template CHUNKED_VERSION<DIMCHK>,NS...>;

    template < int CAT >
    using CHUNKED_VARS = typename F::template CHUNKED_VARS<CAT>;

    template < int CAT >
    using NOTCHUNKED_VARS = typename F::template NOTCHUNKED_VARS<CAT>;

    template < typename TYPE >
    static DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
	using OpScal = typename OP<F,NS...>::template Operation_Scalar<TYPE>;
  	VectApply < OpScal, DIM > (out, outF);
    }

};
  
}
