#pragma once

#include <sstream>
#include "core/autodiff/BinaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pack/CondType.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class,class> class OP, class F, class G >
struct VectorizedScalarBinaryOp : BinaryOp<OP, F, G> {
	
    static const int DIM = ::std::max(F::DIM,G::DIM);
	
	static_assert( ((F::DIM==DIM)||(F::DIM==1)) &&
				   ((G::DIM==DIM)||(G::DIM==1)) , "Incompatible dimensions for vectorized scalar binary operation.");

    static const bool IS_CHUNKABLE = F::IS_CHUNKABLE && G::IS_CHUNKABLE;

    template < int DIMCHK >
    using CHUNKED_VERSION = OP < CondType< F, typename F::template CHUNKED_VERSION<DIMCHK>, F::DIM==1 > , 
				CondType< G, typename G::template CHUNKED_VERSION<DIMCHK>, G::DIM==1 >  >;

    template < int CAT >
    using CHUNKED_VARS = MergePacks < CondType < univpack<> , typename F::template CHUNKED_VARS<CAT>, F::DIM==1 >,
									  CondType < univpack<> , typename G::template CHUNKED_VARS<CAT>, G::DIM==1 > >;

    template < int CAT >
    using NOTCHUNKED_VARS = MergePacks < CondType < typename F::template VARS<CAT> , typename F::template NOTCHUNKED_VARS<CAT>, F::DIM==1 >,
									  	 CondType < typename G::template VARS<CAT> , typename G::template NOTCHUNKED_VARS<CAT>, G::DIM==1 > >;

    template < typename TYPE >
    static DEVICE INLINE void Operation(TYPE *out, TYPE *outF, TYPE *outG) {
		using OpScal = typename OP<F,G>::template Operation_Scalar<TYPE>;
  	    VectApply < OpScal, DIM, F::DIM, G::DIM > (out, outF, outG);
    }

};
  
}
