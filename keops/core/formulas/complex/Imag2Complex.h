#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/ComplexImag.h"

#include "core/pre_headers.h"

namespace keops {

template<class F> struct ComplexImag;

/////////////////////////////////////////////////////////////////////////
////      Imag2Complex                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct Imag2Complex: UnaryOp<Imag2Complex, F> {

  static const int DIM = 2 * F::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "Imag2Complex"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < DIM; i+=2) {
            out[i] = cast_to<TYPE>(0.0f);
            out[i+1] = inF[i/2];
        }
    }

    template<class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, ComplexImag<GRADIN>>;

};

#define Imag2Complex(f) KeopsNS<Imag2Complex<decltype(InvKeopsNS(f))>>()

}
