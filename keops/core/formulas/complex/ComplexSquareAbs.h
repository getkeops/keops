#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/ComplexReal.h"
#include "core/formulas/complex/ComplexMult.h"

#include "core/utils/keops_math.h"
#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexSquareAbs                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct ComplexSquareAbs: UnaryOp<ComplexSquareAbs, F> {

  static_assert(F::DIM % 2 == 0, "Dimension of F must be even");

  static const int DIM = F::DIM / 2;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexSquareAbs"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            out[i/2] = inF[i] * inF[i] + inF[i+1] * inF[i+1];
        }
    }
    
    using AltFormula = ComplexReal < ComplexMult < F , Conj<F> > > ;

    template<class V, class GRADIN>
    using DiffT = typename AltFormula::template DiffT<V, GRADIN>;

};

#define ComplexSquareAbs(f) KeopsNS<ComplexSquareAbs<decltype(InvKeopsNS(f))>>()

}
