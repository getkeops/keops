#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"

#include "core/formulas/constants/Zero.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Pow.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/IntInv.h"

namespace keops {

//////////////////////////////////////////////////////////////
////       INVERSE SQUARE ROOT : Rsqrt< F >               ////
//////////////////////////////////////////////////////////////

template<class F> struct Rsqrt_Alias;
template<class F> using Rsqrt = typename Rsqrt_Alias<F>::type;


template<class F>
struct Rsqrt_Impl : VectorizedScalarUnaryOp<Rsqrt_Impl, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Rsqrt"; }

  template < typename TYPE >
  static DEVICE INLINE void Operation(TYPE *out, TYPE *outF) {
    #pragma unroll
    for (int k = 0; k < F::DIM; k++)
      if (outF[k] == 0.0f)
        out[k] = 0.0f;
      else
        out[k] = keops_rsqrt(outF[k]);
  }

#if USE_HALF && GPU_ON
  static DEVICE INLINE void Operation(half2 *out, half2 *outF) {
    #pragma unroll
    for (int k = 0; k < F::DIM; k++) {
      half2 cond = __heq2(outF[k],__float2half2_rn(0.0f));
      out[k] = h2rsqrt(outF[k]+cond) * (__float2half2_rn(1.0f)-cond);
    }
  }
#endif

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
