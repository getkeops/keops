#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/ComplexReal.h"
#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/maths/ComplexMult.h"

#include "core/utils/keops_math.h"
#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexAbs                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct ComplexAbs: UnaryOp<ComplexAbs, F> {

  static_assert(F::DIM % 2 == 0, "Dimension of F must be even");

  static const int DIM = F::DIM / 2;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexAbs"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            out[i/2] = keops_sqrt(inF[i] * inF[i] + inF[i+1] * inF[i+1]);
        }
    }
    
    using AltFormula = Sqrt < ComplexReal < ComplexMult < F , Conj<F> > > >;

    template<class V, class GRADIN>
    using DiffT = typename AltFormula::template DiffT<V, GRADIN>;

};

#define ComplexAbs(f) KeopsNS<ComplexAbs<decltype(InvKeopsNS(f))>>()

}
