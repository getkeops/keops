#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/ComplexReal.h"

#include "core/pre_headers.h"

namespace keops {

template<class F> struct ComplexReal;

/////////////////////////////////////////////////////////////////////////
////      Real2Complex                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct Real2Complex: UnaryOp<Real2Complex, F> {

  static const int DIM = 2 * F::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "Real2Complex"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < DIM; i+=2) {
            out[i] = inF[i/2];
            out[i+1] = cast_to<TYPE>(0.0f);;
        }
    }

    template<class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, ComplexReal<GRADIN>>;

};

#define Real2Complex(f) KeopsNS<Real2Complex<decltype(InvKeopsNS(f))>>()

}
