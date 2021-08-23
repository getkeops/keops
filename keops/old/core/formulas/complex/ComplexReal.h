#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/Real2Complex.h"

#include "core/pre_headers.h"

namespace keops {

template<class F> struct Real2Complex;

/////////////////////////////////////////////////////////////////////////
////      ComplexReal                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct ComplexReal: UnaryOp<ComplexReal, F> {

  static_assert(F::DIM % 2 == 0, "Dimension of F must be even");

  static const int DIM = F::DIM / 2;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexReal"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            out[i/2] = inF[i];
        }
    }

    template<class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, Real2Complex<GRADIN>>;

};

#define ComplexReal(f) KeopsNS<ComplexReal<decltype(InvKeopsNS(f))>>()

}
