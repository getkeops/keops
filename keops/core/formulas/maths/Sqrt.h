#pragma once

#include <sstream>

#include "core/autodiff.h"
#include "core/formulas/constants.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/IntInv.h"
#include "core/formulas/maths/Rsqrt.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////       SQUARE ROOT : Sqrt< F >                        ////
//////////////////////////////////////////////////////////////

template<class F> struct Sqrt_Impl;
template<class F> struct Sqrt_Alias;
template<class F> using Sqrt = typename Sqrt_Alias<F>::type;

template<class F>
struct Sqrt_Impl : UnaryOp<Sqrt_Impl, F> {
  static const int DIM = F::DIM;

  static void PrintIdString(std::stringstream &str) {
    str << "Sqrt";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = sqrt(outF[k]);
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Scal<IntInv<2>, Rsqrt<F>>, GRADIN>>;
};

// Simplification rule

// base class, redirects to implementation
template<class F>
struct Sqrt_Alias {
  using type = Sqrt_Impl<F>;
};

// Sqrt(0) = 0
template<int DIM>
struct Sqrt_Alias<Zero<DIM>> {
using type = Zero<DIM>;
};

#define Sqrt(f) KeopsNS<Sqrt<decltype(InvKeopsNS(f))>>()

}
