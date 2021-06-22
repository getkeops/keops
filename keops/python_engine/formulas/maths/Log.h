#pragma once

#include <sstream>
#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Inv.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             LOGARITHM : Log< F >                     ////
//////////////////////////////////////////////////////////////

template<class F>
struct Log : VectorizedScalarUnaryOp<Log, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Log"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_log(outF);
    }
  };

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Inv<F>, GRADIN>>;
};

#define Log(f) KeopsNS<Log<decltype(InvKeopsNS(f))>>()

}
