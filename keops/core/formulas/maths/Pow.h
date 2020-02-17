#pragma once

#include <sstream>
#include <cmath>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Inv.h"

#include "core/formulas/maths/Powf.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Pow< F, M >             ////
//////////////////////////////////////////////////////////////

#if USE_HALF

// there is no power operator available yet for half2...
// so we use recursive definition F^M = F^(M/2) * F^(M/2).
template<class F, int M>
struct Pow_Impl {
  using left = typename Pow_Impl<F,M/2>::type;
  using right = typename Pow_Impl<F,M-M/2>::type;
  using type = typename Mult_Alias<left, right>::type;
};

template<class F>
struct Pow_Impl<F,1> {
  using type = F;
};

template<class F>
struct Pow_Impl<F,-1> {
  using type = Inv<F>;
};

template<class F>
struct Pow_Impl<F,0> {
  using type = IntConstant<1>;
};

template<class F, int M>
using Pow = typename Pow_Impl<F,M>::type;

#else

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

#endif

#define Pow(f,M) KeopsNS<Pow<decltype(InvKeopsNS(f)),M>>()

}
