#pragma once

#include <sstream>
#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Log.h"
#include "core/formulas/constants/IntConst.h"

namespace keops {

//////////////////////////////////////////////////////////////
////        X*LOG(X) FUNCTION : XLogX< F >                ////
//////////////////////////////////////////////////////////////

template<class F>
struct XLogX : VectorizedScalarUnaryOp<XLogX, F> {

  static void PrintIdString(::std::stringstream &str) { str << "XLogX"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_xlogx(outF);
    }
  };

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Add<Log<F>,IntConstant<1>>, GRADIN>>;
};

#define XLogX(f) KeopsNS<XLogX<decltype(InvKeopsNS(f))>>()

}
