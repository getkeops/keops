#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarBinaryOp.h"



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

  // d/dx atan2(y, x) = -y /(x**2 + y**2) and d/dy atan2(y, x) = x /(x**2 + y**2)
  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V,GRADIN>;

};

#define Atan2(f,g) KeopsNS<Atan2<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
