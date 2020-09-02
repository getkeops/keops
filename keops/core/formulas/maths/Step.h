#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/constants/Zero.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             STEP : Step< F >                         ////
//////////////////////////////////////////////////////////////

template < class F >
struct Step : VectorizedScalarUnaryOp< Step, F > {

  static void PrintIdString(::std::stringstream &str) { str << "Step"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_step(outF);
    }
  };

  template < class V, class GRADIN >
  using DiffT = Zero< V::DIM >;
};

#define Step(f) KeopsNS<Step<decltype(InvKeopsNS(f))>>()

}
