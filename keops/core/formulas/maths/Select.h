#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/maths/SelectT.h"
#include "core/pre_headers.h"

namespace keops {

template< class F, class G, int D, int FFDIM >
struct SelectT;

//////////////////////////////////////////////////////////////
////     VECTOR SELECTION : Select<FF,G,DIM,FDIM>         ////
//////////////////////////////////////////////////////////////

template< class FF, class G, int D, int FDIM >
struct Select : BinaryOp< Select, FF, G, D, FDIM > {
  static const int DIM = FDIM;

  static_assert(FF::DIM == DIM * D, "Selects should pick values in a vector of size 'D * F::DIM'.");
  static_assert(G::DIM == 1, "Select only supports scalar indexing.");

  static void PrintIdString(::std::stringstream &str) {
    str << " select ";
  }

  static HOST_DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outFF, __TYPE__ *outG) {

    int index = round(outG[0]);  // read the value of "G"

    // Hopefully, the compiler will handle the branching efficiently...
    if (0 <= index && index < D) {  // Boundary conditions.
      for (int k = 0; k < DIM; k++)
        out[k] = outFF[index * FDIM + k];
    }
    else {  // If we're outside of the [0,D) range, let's just return 0
      for (int k = 0; k < DIM; k++)
        out[k] = 0.0f;
    }
  }

  template< class V, class GRADIN >
  using DiffTFF = typename FF::template DiffT<V, GRADIN>;

  template< class V, class GRADIN >
  using DiffT = DiffTFF< V, SelectT< GRADIN, G, D, FF::DIM > >;
};

#define Select(ff,g,d,fd) KeopsNS<Select<decltype(InvKeopsNS(ff)),decltype(InvKeopsNS(g)),d,fd>>()

}
