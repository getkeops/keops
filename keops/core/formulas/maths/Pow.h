#pragma once

#include <sstream>
#include <cmath>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Pow< F, M >             ////
//////////////////////////////////////////////////////////////

template<class F, int M>
struct Pow : UnaryOp<Pow, F, M> {

  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "Pow";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = pow(outF[k], M);
  }

  // [\partial_V F^M].gradin  =  M * (F^(M-1)) * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = Scal<IntConstant<M>, DiffTF<V, Mult<Pow<F, M - 1>, GRADIN>>>;

};

#define Pow(f,M) KeopsNS<Pow<decltype(InvKeopsNS(f)),M>>()

}
