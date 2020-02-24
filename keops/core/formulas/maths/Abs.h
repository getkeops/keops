#pragma once

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Sign.h"
#include "core/formulas/maths/Mult.h"

namespace keops {
	
//////////////////////////////////////////////////////////////
////           ABSOLUTE VALUE : Abs< F >                  ////
//////////////////////////////////////////////////////////////

template<class F>
struct Abs : VectorizedScalarUnaryOp<Abs, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Abs"; }

  template < typename TYPE > struct Operation_Scalar {
  	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
      	  out = keops_abs(outF);
	}
  };
  
  // [\partial_V abs(F)].gradin = sign(F) * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Sign<F>, GRADIN>>;

};

#define Abs(f) KeopsNS<Abs<decltype(InvKeopsNS(f))>>()
}
