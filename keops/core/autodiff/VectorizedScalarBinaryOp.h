#pragma once

#include <sstream>
#include "core/autodiff/BinaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class,class> class OP, class F, class G >
struct VectorizedScalarBinaryOp : BinaryOp<OP, F, G> {
	
    static const int DIM = F::DIM;

    template < typename TYPE >
    static DEVICE INLINE void Operation(TYPE *out, TYPE *outF, TYPE *outG) {
		using OpScal = typename OP<F,G>::template Operation_Scalar<TYPE>;
  	    VectApply < OpScal, DIM > (out, outF, outG);
    }

};
  
}
