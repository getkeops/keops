#pragma once

#include <sstream>

#include "core/autodiff.h"
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

  static void PrintIdString(std::stringstream &str) {
    str << "Log";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = log(outF[k]);
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Inv<F>, GRADIN>>;
};

#define Log(f) KeopsNS<Log<decltype(InvKeopsNS(f))>>()

}
