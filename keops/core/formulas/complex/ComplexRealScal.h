#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/Real2Complex.h"
#include "core/formulas/complex/ComplexScal.h"

#include "core/pre_headers.h"

namespace keops {

/////////////////////////////////////////////////////////////////////////
////      ComplexRealScal                          ////
/////////////////////////////////////////////////////////////////////////

template<class F, class G>
struct ComplexRealScal_Impl: BinaryOp<ComplexRealScal_Impl, F, G> {

  static_assert(F::DIM == 1, "Dimension of F must be 1");
  static_assert(G::DIM%2==0, "Dimension of G must be even");

  static const int DIM = G::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexRealScal"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF, TYPE *inG) {
        #pragma unroll
        for (int i = 0; i < DIM; i+=2) {
            out[i] = *inF * inG[i];
            out[i+1] = *inF * inG[i+1];
        }
    }

    using AltFormula = ComplexScal < Real2Complex<F> , G >;

    template<class V, class GRADIN>
    using DiffT = typename AltFormula::template DiffT<V,GRADIN>;

};

template<class F, class G>
using ComplexRealScal = CondType < ComplexRealScal_Impl<F,G> , ComplexRealScal_Impl<G,F> , F::DIM==1 >;

#define ComplexRealScal(f,g) KeopsNS<ComplexRealScal<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
