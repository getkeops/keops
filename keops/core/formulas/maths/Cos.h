#pragma once

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Sin.h"
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Mult.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                 COSINE :  Cos< F >                   ////
//////////////////////////////////////////////////////////////


template<class F>
struct Sin;


template<class F>
struct Cos : VectorizedScalarUnaryOp<Cos, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Cos"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_cos(outF);
    }
  };

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Minus<Mult<Sin<F>, GRADIN>>>;

};

#define Cos(f) KeopsNS<Cos<decltype(InvKeopsNS(f))>>()

}
