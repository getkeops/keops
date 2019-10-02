#pragma once

#include <sstream>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/constants/Zero.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Pow.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/IntInv.h"

#include "core/pre_headers.h"

namespace keops {

//////////////////////////////////////////////////////////////
////       INVERSE SQUARE ROOT : Rsqrt< F >               ////
//////////////////////////////////////////////////////////////

template<class F> struct Rsqrt_Alias;
template<class F> using Rsqrt = typename Rsqrt_Alias<F>::type;


template<class F>
struct Rsqrt_Impl : UnaryOp<Rsqrt_Impl, F> {
  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "Rsqrt";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
#pragma unroll
    for (int k = 0; k < DIM; k++) {
      if (outF[k] == 0) {
        out[k] = 0;  // warning !! value should be Inf at 0 but we put 0 instead. This is intentional...
      }
      else {
#ifdef __CUDA_ARCH__    // check if we are compiling a device code
#if USE_DOUBLE
        out[k] = rsqrt(outF[k]);
#else
        out[k] = rsqrtf(outF[k]);
#endif
#else
        out[k] = 1.0 / sqrt(outF[k]); // should use specific rsqrt implementation for cpp ..
#endif
      }
    }
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = DiffTF<V, Mult<Scal<IntInv<-2>, Pow<Rsqrt<F>, 3>>, GRADIN>>;
};

// base class, redirects to implementation
template<class F>
struct Rsqrt_Alias {
  using type = Rsqrt_Impl<F>;
};

// Rsqrt(0) = 0   // warning !! Rsqrt(0) should be Inf but we put 0 instead. This is intentional...
template<int DIM>
struct Rsqrt_Alias<Zero<DIM>> {
using type = Zero<DIM>;
};

#define Rsqrt(f) KeopsNS<Rsqrt<decltype(InvKeopsNS(f))>>()
}
