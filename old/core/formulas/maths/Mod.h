#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarTernaryOp.h"



namespace keops {

//////////////////////////////////////////////////////////////
////                 MODULO :  Mod< F, A, B >             ////
//////////////////////////////////////////////////////////////


template<class F, class G, class H>
struct Mod : VectorizedScalarTernaryOp<Mod, F, G, H> {
  
  static void PrintIdString(::std::stringstream &str) { str << "Mod"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	  DEVICE INLINE void operator()(TYPE &out, TYPE &outF, TYPE &outG, TYPE &outH) {
		  out = keops_mod(outF, outG, outH);
	  }
  };

  template<class V, class GRADIN>
  using DiffT = typename F::template DiffT<V,GRADIN>;

};

#define Mod(f,g,h) KeopsNS<Mod<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g)),decltype(InvKeopsNS(h))>>()

}
