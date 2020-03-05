#pragma once

#include <sstream>
#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Square.h"

namespace keops {

//////////////////////////////////////////////////////////////
////      INVERSE : Inv<F>                                ////
//////////////////////////////////////////////////////////////

//template < class F >
//using Inv = Pow<F,-1>;

template<class F>
struct Inv : VectorizedScalarUnaryOp<Inv, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Inv"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_rcp(outF);
    }
  };

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  // [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = Scal<IntConstant<-1>, DiffTF<V, Mult<Square<Inv<F>>, GRADIN>>>;

};

#define Inv(f) KeopsNS<Inv<decltype(InvKeopsNS(f))>>()

}
