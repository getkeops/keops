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

////////////////////////////////////////////////////////////////////
////         SINXDIVX :  SinXDivX < F >                   ////
////////////////////////////////////////////////////////////////////


template<class F>
struct SinXDivX : VectorizedScalarUnaryOp<SinXDivX, F> {

  static void PrintIdString(::std::stringstream &str) { str << "SinXDivX"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_sinxdivx(outF);
    }
  };
 
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Subtract<Divide<Cos<F>, F>, Divide<Sin<F>,Square<F>>>, GRADIN>>;
};

#define SinXDivX(f) KeopsNS<SinXDivX<decltype(InvKeopsNS(f))>>()

}
