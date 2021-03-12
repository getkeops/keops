#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"

#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      Conj : complex conjugate         ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct Conj: UnaryOp<Conj, F> {

  static_assert(F::DIM % 2 == 0, "Dimension of F must be even");

  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "Conj"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            out[i] = inF[i];
            out[i+1] = -inF[i+1];
        }
    }

    template<class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, Conj<GRADIN>>;

};

#define Conj(f) KeopsNS<Conj<decltype(InvKeopsNS(f))>>()

}
