#pragma once

#include <sstream>
#include <cmath>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Inv.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             LOGARITHM : Log< F >                     ////
//////////////////////////////////////////////////////////////

template<class F>
struct Log : UnaryOp<Log, F> {
  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "Log";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++) {
#if USE_DOUBLE
      out[k] = log(outF[k]);
#else
      out[k] = logf(outF[k]);
#endif
    }
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Inv<F>, GRADIN>>;
};

#define Log(f) KeopsNS<Log<decltype(InvKeopsNS(f))>>()

}
