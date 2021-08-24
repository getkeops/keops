#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/complex/Conj.h"

#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexMult                           ////
/////////////////////////////////////////////////////////////////////////

template<class F, class G>
struct ComplexMult: BinaryOp<ComplexMult, F, G> {

  static_assert(F::DIM % 2 == 0, "Dimension of F must be even");
  static_assert(F::DIM==G::DIM, "Dimensions of F and G must be equal");

  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexMult"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF, TYPE *inG) {
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            out[i] = inF[i] * inG[i] - inF[i+1] * inG[i+1];
            out[i+1] = inF[i] * inG[i+1] + inF[i+1] * inG[i];
        }
    }

    template<class V, class GRADIN>
    using DiffTF = typename F::template DiffT<V, GRADIN>;

    template<class V, class GRADIN>
    using DiffTG = typename G::template DiffT<V, GRADIN>;

    template<class V, class GRADIN>
    using DiffT = Add<DiffTF<V, ComplexMult<Conj<G>, GRADIN>>, DiffTG<V, ComplexMult<Conj<F>, GRADIN>>>;

};

#define ComplexMult(f,g) KeopsNS<ComplexMult<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
