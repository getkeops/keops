#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Step.h"



namespace keops {

//////////////////////////////////////////////////////////////
////             RELU : ReLU< F >                         ////
//////////////////////////////////////////////////////////////

template<class F>
struct ReLU : VectorizedScalarUnaryOp<ReLU, F> {

  static void PrintIdString(::std::stringstream &str) { str << "ReLU"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_relu(outF);
    }
  };

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Step<F>, GRADIN>>;
};

#define ReLU(f) KeopsNS<ReLU<decltype(InvKeopsNS(f))>>()

}
