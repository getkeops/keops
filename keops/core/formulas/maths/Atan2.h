#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarBinaryOp.h"

#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Square.h"
#include "core/formulas/maths/Divide.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Subtract.h"


namespace keops {

//////////////////////////////////////////////////////////////
////                 ATAN2 :  Atan2< F, G >               ////
//////////////////////////////////////////////////////////////


template<class F, class G>
struct Atan2 : VectorizedScalarBinaryOp<Atan2, F, G> {
  
  static void PrintIdString(::std::stringstream &str) { str << "Atan2"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	  DEVICE INLINE void operator()(TYPE &out, TYPE &outF, TYPE &outG) {
		  out = keops_atan2(outF, outG);
	  }
  };

  // [ \partial_V Atan2(F, G) ] . gradin = [ (G / F^2 + G^2) . \partial_V F ] . gradin  - [ (F / F^2 + G^2) . \partial_V G ] . gradin
  template < class V, class GRADIN >
  using partial_F = typename F::template DiffT< V, Mult< Divide<G, Add<Square<F>, Square<G>> >, GRADIN> >;
  template < class V, class GRADIN >
  using partial_G = typename G::template DiffT< V, Mult< Divide<F, Add<Square<F>, Square<G>> >, GRADIN> >;

  template < class V, class GRADIN >
  using DiffT = Subtract<partial_F<V, GRADIN>, partial_G<V, GRADIN>>;

};

#define Atan2(f,g) KeopsNS<Atan2<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()


}
