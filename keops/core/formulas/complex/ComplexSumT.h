#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/ComplexSum.h"

#include "core/pre_headers.h"

namespace keops {

template<class F> struct ComplexSum;

/////////////////////////////////////////////////////////////////////////
////      adjoint of ComplexSum                           ////
/////////////////////////////////////////////////////////////////////////

template<class F, int D>
struct ComplexSumT: UnaryOp<ComplexSumT, F, D> {

  static_assert(F::DIM == 2, "Dimension of F must be 2");
  static_assert(D%2 == 0, "D must be even");

  static const int DIM = D;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexSumT"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < D; i+=2) {
            out[i] = inF[0];
            out[i+1] = inF[1];
        }
    }

    template<class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, ComplexSum<GRADIN>>;

};

#define ComplexSumT(f) KeopsNS<ComplexSumT<decltype(InvKeopsNS(f))>>()

}
