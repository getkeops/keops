#pragma once

#include <sstream>
#include "core/autodiff/BinaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class,class> class OP, class F, class G >
struct VectorizedScalarBinaryOp : BinaryOp<OP, F, G> {
	
    static const int DIM = F::DIM;

    static const bool IS_CHUNKABLE = F::IS_CHUNKABLE && G::IS_CHUNKABLE;

    template < int DIMCHK >
    using CHUNKED_VERSION = OP<typename F::template CHUNKED_VERSION<DIMCHK>,typename G::template CHUNKED_VERSION<DIMCHK>>;

    template < int CAT >
    using CHUNKED_VARS = MergePacks<typename F::template CHUNKED_VARS<CAT>,typename G::template CHUNKED_VARS<CAT>>;

    template < int CAT >
    using NOTCHUNKED_VARS = MergePacks<typename F::template NOTCHUNKED_VARS<CAT>,typename G::template NOTCHUNKED_VARS<CAT>>;

    template < typename TYPE >
    static DEVICE INLINE void Operation(TYPE *out, TYPE *outF, TYPE *outG) {
		using OpScal = typename OP<F,G>::template Operation_Scalar<TYPE>;
  	    VectApply < OpScal, DIM > (out, outF, outG);
    }

};
  
}
