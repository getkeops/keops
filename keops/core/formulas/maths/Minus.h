#pragma once

#include <sstream>

#include "core/autodiff/VectorizedScalarUnaryOp.h"

namespace keops {

//////////////////////////////////////////////////////////////
////               MINUS OPERATOR : Minus< F >            ////
//////////////////////////////////////////////////////////////

template<class F>
struct Minus : VectorizedScalarUnaryOp<Minus, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Minus"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = - outF;
    }
  };

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Minus<GRADIN>>;

};

template < class F >
KeopsNS<Minus<F>> operator-(KeopsNS<F> f) {
  return KeopsNS<Minus<F>>();
}
#define Minus(f) KeopsNS<Minus<decltype(InvKeopsNS(f))>>()


}
