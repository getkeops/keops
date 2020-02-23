#pragma once

#include <sstream>
#include "core/autodiff/UnaryOp.h"
#include "core/utils/TypesUtils.h"
#include "core/pre_headers.h"

namespace keops {

template< template<class> class OP, class F >
struct VectorizedScalarOp : UnaryOp<OP, F> {
	
    static const int DIM = F::DIM;

    template < typename TYPE >
    static DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
		using OpScal = typename OP<F>::template Operation_Scalar<TYPE>;
  	    VectApply < OpScal, DIM > (out, outF);
    }

};
  
}
