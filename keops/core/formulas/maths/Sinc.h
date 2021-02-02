#pragma once

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Cos.h"
#include "core/formulas/maths/Sin.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Divide.h"
#include "core/formulas/maths/Square.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                 SINC :  Sinc < F >                   ////
//////////////////////////////////////////////////////////////


template<class F>
struct Sinc : VectorizedScalarUnaryOp<Sinc, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Sinc"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_sinc(outF);
    }
  };
 
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Subtract<Divide<Cos<F>, F>, Divide<Sin<F>,Square<F>>>, GRADIN>>;
};

#define Sinc(f) KeopsNS<Sinc<decltype(InvKeopsNS(f))>>()

}
