#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/constants/Zero.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             SIGN : Sign< F >                         ////
//////////////////////////////////////////////////////////////

template<class F>
struct Sign : VectorizedScalarUnaryOp<Sign, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Sign"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_sign(outF);
    }
  };

  template<class V, class GRADIN>
  using DiffT = Zero<V::DIM>;
};

#define Sign(f) KeopsNS<Sign<decltype(InvKeopsNS(f))>>()

}
