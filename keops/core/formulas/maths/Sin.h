#pragma once

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Cos.h"
#include "core/formulas/maths/Mult.h"

namespace keops {

//////////////////////////////////////////////////////////////
////                  SINE :  Sin< F >                    ////
//////////////////////////////////////////////////////////////

template<class F>
struct Cos;

template<class F>
struct Sin : VectorizedScalarUnaryOp<Sin, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Sin"; }

  template < typename TYPE > struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_sin(outF);
    }
  };

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Cos<F>, GRADIN>>;

};

#define Sin(f) KeopsNS<Sin<decltype(InvKeopsNS(f))>>()

}
