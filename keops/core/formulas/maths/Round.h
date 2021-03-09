#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/constants/Zero.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             ROUND : Round< F, D >                    ////
//////////////////////////////////////////////////////////////

template < class F, int D >
struct Round : VectorizedScalarUnaryOp< Round, F, D > {

  static void PrintIdString(::std::stringstream &str) { str << "Round"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_round(outF, D);
    }
  };

  template < class V, class GRADIN >
  using DiffT = Zero< V::DIM >;
};

#define Round(f,d) KeopsNS<Round<decltype(InvKeopsNS(f)),d>>()

}
