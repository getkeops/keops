#pragma once

#include <assert.h>
#include <sstream>

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/norms/SqNorm2.h"

namespace keops {

//////////////////////////////////////////////////////////////
////          ISOTROPIC NORM :   SqNormIso< S,A >         ////
//////////////////////////////////////////////////////////////

// Isotropic norm, if S is a scalar:
// SqNormIso<S,A> = S * <A,A> = S * sum_i a_i*a_i
template<class FS, class FA>
struct SqNormIso : BinaryOp<SqNormIso, FS, FA> {
  // Output dimension = 1, provided that FS::DIM = 1
  static const int DIMIN = FA::DIM;
  static_assert(FS::DIM == 1, "Isotropic square norm expects a scalar parameter.");
  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) {
    str << "<SqNormIso>";
  }

  static DEVICE INLINE
  void Operation(__TYPE__ *out, __TYPE__ *outS, __TYPE__ *outA) {
    *out = 0;
    for (int k = 0; k < DIMIN; k++)
      *out += outA[k] * outA[k];
    *out *= *outS;
  }

  // S*<A,A> is scalar-valued, so that gradin is necessarily a scalar.
  // [\partial_V S*<A,A>].gradin = gradin * ( 2*S*[\partial_V A].A + [\partial_V S].<A,A> )
  template<class V, class GRADIN>
  using DiffT = Scal <GRADIN,
  Add<Scal< Scal< IntConstant < 2>, FS>, typename FA::template DiffT<V, FA> >,
  typename FS::template DiffT<V, SqNorm2 < FA> >
  >
  >;
};

}
