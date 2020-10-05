#pragma once

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Mult.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             EXPONENTIAL : Exp< F >                   ////
//////////////////////////////////////////////////////////////

template<class F>
struct Exp : VectorizedScalarUnaryOp<Exp, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Exp"; }

  template < typename TYPE > 
  struct Operation_Scalar {
    DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
  	  out = keops_exp(outF);
    }
  };

  // [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V, Mult<Exp<F>, GRADIN>>;

};

#define Exp(f) KeopsNS<Exp<decltype(InvKeopsNS(f))>>()

}
