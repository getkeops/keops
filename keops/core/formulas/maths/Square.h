#pragma once

#include <sstream>

#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"

#include "core/pre_headers.h"

namespace keops {


//////////////////////////////////////////////////////////////
////             SQUARED OPERATOR : Square< F >           ////
//////////////////////////////////////////////////////////////

//template < class F >
//using Square = Pow<F,2>;

template<class F>
struct Square : VectorizedScalarUnaryOp<Square, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Sq"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = outF * outF;
    }
  };

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  // [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffT = Scal<IntConstant<2>, DiffTF<V, Mult<F, GRADIN>>>;

};

#define Square(f) KeopsNS<Square<decltype(InvKeopsNS(f))>>()

}
