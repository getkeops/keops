#pragma once

#include <sstream>
#include <cmath>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Log.h"
#include "core/formulas/constants/IntConst.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////        X*LOG(X) FUNCTION : XLogX< F >                ////
//////////////////////////////////////////////////////////////

template<class F>
struct XLogX : UnaryOp<XLogX, F> {
  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "XLogX";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++) {
#if USE_DOUBLE
      out[k] = outF[k] ? outF[k]*log(outF[k]) : 0.0;
#else
      out[k] = outF[k] ? outF[k]*logf(outF[k]) : 0.0;
#endif
    }
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Add<Log<F>,IntConstant<1>>, GRADIN>>;
};

#define XLogX(f) KeopsNS<XLogX<decltype(InvKeopsNS(f))>>()

}
