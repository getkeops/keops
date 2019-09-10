#pragma once

#include <assert.h>
#include <sstream>

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"

namespace keops {

// Anisotropic (but diagonal) norm, if S::DIM == A::DIM:
// SqNormDiag<S,A> = sum_i s_i*a_i*a_i
template < class FS, class FA >
struct SqNormDiag : BinaryOp<SqNormDiag,FS,FA> {
  // Output dimension = 1, provided that FS::DIM = FA::DIM
  static const int DIMIN = FA::DIM;
  static_assert(FS::DIM==FA::DIM,"Diagonal square norm expects a vector of parameters of dimension FA::DIM.");
  static const int DIM = 1;

  static void PrintIdString(::std::stringstream& str) {
    str << "<SqNormDiag>";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outS, __TYPE__ *outA) {
    *out = 0;
    for(int k=0; k<DIMIN; k++)
      *out += outS[k]*outA[k]*outA[k];
  }

  // sum_i s_i*a_i*a_i is scalar-valued, so that gradin is necessarily a scalar.
  // [\partial_V ...].gradin = gradin * ( 2*[\partial_V A].(S*A) + [\partial_V S].(A*A) )
  template < class V, class GRADIN >
  using DiffT = Scal < GRADIN,
      Add< Scal< IntConstant<2>, typename FA::template DiffT<V,Mult<FS,FA>> >,
  typename FS::template DiffT<V, Mult<FA,FA> >
  >
  >;
};

}
