#pragma once

#include <sstream>
#include <cmath>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/constants/Zero.h"
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
struct Sqrt_Impl : VectorizedScalarUnaryOp<Sqrt_Impl, F> {

  static void PrintIdString(::std::stringstream &str) { str << "Sqrt"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_sqrt(outF);
    }
  };

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
