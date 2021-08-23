#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/ReLu.h"
#include "core/formulas/maths/DiffClampInt.h"
#include "core/formulas/constants/IntConst.h"


namespace keops {

//////////////////////////////////////////////////////////////
////             CLAMPINT : ClampInt< F, A, B >           ////
//////////////////////////////////////////////////////////////

// ClampInt(x,a,b) = a if x<a, x if a<=x<=b, b if b<x 
// N.B. same as Clamp but a and b are fixed integers.
// ClampInt may be faster than Clamp because we avoid the transfer
// of A and B in memory.

template<class F, int A, int B>
struct ClampInt : VectorizedScalarUnaryOp<ClampInt, F, A, B> {

  static void PrintIdString(::std::stringstream &str) { str << "ClampInt"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_clampint(outF, A, B);
    }
  };

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult< DiffClampInt<F,A,B>, GRADIN > >;

};

#define ClampInt(f,A,B) KeopsNS<ClampInt<decltype(InvKeopsNS(f)),A,B>>()

}
