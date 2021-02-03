#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarBinaryOp.h"
#include "core/formulas/constants/Zero.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             ROUND : Round< F, G >                    ////
//////////////////////////////////////////////////////////////

template < class F, class G >
struct Round : VectorizedScalarBinaryOp< Round, F, G > {

  static void PrintIdString(::std::stringstream &str) { str << "Round"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF, TYPE &outG) {
    	  out = keops_round(outF, outG);
    }
  };

  template < class V, class GRADIN >
  using DiffT = Zero< V::DIM >;
};

#define Round(f,g) KeopsNS<Round<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
