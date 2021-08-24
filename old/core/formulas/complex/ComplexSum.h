#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/ComplexSumT.h"

#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexSum                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct ComplexSum: UnaryOp<ComplexSum, F> {

  static_assert(F::DIM % 2 == 0, "Dimension of F must be even");

  static const int DIM = 2;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexSum"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        out[0] = cast_to<TYPE>(0.0f);
        out[1] = out[0];
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            out[0] += inF[i];
            out[1] += inF[i+1];
        }
    }


    template<class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, ComplexSumT<GRADIN, F::DIM>>;

};

#define ComplexSum(f) KeopsNS<ComplexSum<decltype(InvKeopsNS(f))>>()

}
