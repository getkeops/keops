#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarTernaryOp.h"
#include "core/formulas/maths/ClampInt.h"



namespace keops {

//////////////////////////////////////////////////////////////
////             CLAMP : Clamp< F, G, H >                 ////
//////////////////////////////////////////////////////////////

// Clamp(x,a,b) = a if x<a, x if a<=x<=b, b if b<x 

template<class F,class G,class H>
struct Clamp : VectorizedScalarTernaryOp<Clamp, F, G, H> {
  
  static void PrintIdString(::std::stringstream &str) { str << "Clamp"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	  DEVICE INLINE void operator()(TYPE &out, TYPE &outF, TYPE &outG, TYPE &outH) {
		  out = keops_clamp(outF, outG, outH);
	  }
  };

  // N.B.   Clamp(F,G,H) = Clamp((F-G)/(H-G),0,1) * (H-G) + G
  // We use this fact to avoid writing another custom operation for the gradient.
  // (This may be slower however...)
  
  using Alt_Clamp = Add<ScalOrMult<ClampInt<Divide<Subtract<F,G>,Subtract<H,G>>,0,1>,Subtract<H,G>>,G>;

  template<class V, class GRADIN>
  using DiffT = typename Alt_Clamp::template DiffT<V,GRADIN>;
  
};	

#define Clamp(f,g,h) KeopsNS<Clamp<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g)),decltype(InvKeopsNS(h))>>()

}
