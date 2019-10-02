#pragma once

#include <sstream>
#include <assert.h>

#include "lib/sequences/include/tao/seq/select.hpp"
#include "core/autodiff/BinaryOp.h"
#include "core/formulas/maths/Select.h"
#include "core/pre_headers.h"

namespace keops {

template< class FF, class G, class D, class FDIM >
struct Select;

//////////////////////////////////////////////////////////////
////     VECTOR "INJECTION" : SelectT<F,G,D,FFDIM>        ////
//////////////////////////////////////////////////////////////

// N.B.: D and FDIM are actually integers, but have
//       to be encapsulated as tao::seq objects
//       to fit within the BinaryOp guidelines
template < class F, class G, class D_, class FFDIM_ >
struct SelectT : BinaryOp< SelectT, F, G, D_, FFDIM_> {

  static const int FFDIM = tao::seq::select<0,FFDIM_>::value;
  static const int D     = tao::seq::select<0,D_>::value;

  static const int DIM = FFDIM;

  static_assert(DIM == F::DIM * D, "SelectT should embed a vector of size F::DIM in a vector of size 'F::DIM * D'.");
  static_assert(G::DIM == 1, "SelectT only supports scalar indexing.");

  static void PrintIdString(::std::stringstream& str) {
    str << " selectT ";
  }

  static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF, __TYPE__ *outG) {
    
    int index = round(outG[0]);    // read the value of "G"
    for (int k = 0; k < DIM; k++)  // Fill the output with zeros
      out[k] = 0.0f;

    // Hopefully, the compiler will handle the branching efficiently...
    if (0 <= index && index < D) {  // Boundary conditions.
      for (int k = 0; k < F::DIM; k++)
        out[index * F::DIM + k] = outF[k];
    }
  }

  template < class V, class GRADIN >
  using DiffTF = typename F::template DiffT<V,GRADIN>;

  template < class V, class GRADIN >
  using DiffT = DiffTF<V,Select<GRADIN,G,Ind(D),Ind(F::DIM)>>;
};

#define SelectT(f,g,d,ffd) KeopsNS<SelectT<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g)),d,ffd>>()

}
