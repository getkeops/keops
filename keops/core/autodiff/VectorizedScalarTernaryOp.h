#pragma once

#include <sstream>
#include "core/autodiff/TernaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class,class,class> class OP, class F, class G, class H >
struct VectorizedScalarTernaryOp : TernaryOp<OP, F, G, H> {
	
    static const int DIM = F::DIM;

    static const bool IS_CHUNKABLE = F::IS_CHUNKABLE && G::IS_CHUNKABLE && H::IS_CHUNKABLE;

    template < int DIMCHK >
    using CHUNKED_VERSION = OP< typename F::template CHUNKED_VERSION<DIMCHK>,
				typename G::template CHUNKED_VERSION<DIMCHK>,
				typename H::template CHUNKED_VERSION<DIMCHK> >;

    template < int CAT >
    using CHUNKED_VARS = MergePacks< typename F::template CHUNKED_VARS<CAT>,
							MergePacks< typename G::template CHUNKED_VARS<CAT>,
											typename H::template CHUNKED_VARS<CAT> > >;

    template < int CAT >
    using NOTCHUNKED_VARS = MergePacks< typename F::template NOTCHUNKED_VARS<CAT>,
								MergePacks< typename G::template NOTCHUNKED_VARS<CAT>,
												typename H::template NOTCHUNKED_VARS<CAT> > >;

    template < typename TYPE >
    static DEVICE INLINE void Operation(TYPE *out, TYPE *outF, TYPE *outG, TYPE *outH) {
		using OpScal = typename OP<F,G,H>::template Operation_Scalar<TYPE>;
  	    VectApply < OpScal, DIM > (out, outF, outG, outH);
    }

};
  
}
