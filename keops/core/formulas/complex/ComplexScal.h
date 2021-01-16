#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/complex/Conj.h"

#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexScal                          ////
/////////////////////////////////////////////////////////////////////////

template<class F, class G>
struct ComplexScal_Impl: BinaryOp<ComplexScal_Impl, F, G> {

  static_assert(F::DIM == 2, "Dimension of F must be 2");
  static_assert(G::DIM%2==0, "Dimension of G must be even");

  static const int DIM = G::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexScal"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF, TYPE *inG) {
        #pragma unroll
        for (int i = 0; i < DIM; i+=2) {
            out[i] = inF[0] * inG[i] - inF[1] * inG[i+1];
            out[i+1] = inF[0] * inG[i+1] + inF[1] * inG[i];
        }
    }

    template<class V, class GRADIN>
    using DiffTF = typename F::template DiffT<V, GRADIN>;

    template<class V, class GRADIN>
    using DiffTG = typename G::template DiffT<V, GRADIN>;

    template<class V, class GRADIN>
    using DiffT = Add<DiffTF<V, ComplexSum<ComplexMult<Conj<G>, GRADIN>>>, DiffTG<V, ComplexMult<Conj<F>, GRADIN>>>;

};

template<class F, class G>
using ComplexScal = CondType < ComplexScal_Impl<F,G> , ComplexScal_Impl<G,F> , F::DIM==2 >;

#define ComplexScal(f,g) KeopsNS<ComplexScal<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
