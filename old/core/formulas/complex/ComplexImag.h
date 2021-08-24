#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/Imag2Complex.h"

#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexImag                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct ComplexImag: UnaryOp<ComplexImag, F> {

  static_assert(F::DIM % 2 == 0, "Dimension of F must be even");

  static const int DIM = F::DIM / 2;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexImag"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            out[i/2] = inF[i+1];
        }
    }

    template<class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, Imag2Complex<GRADIN>>;

};

#define ComplexImag(f) KeopsNS<ComplexImag<decltype(InvKeopsNS(f))>>()

}
