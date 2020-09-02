#pragma once

#include <sstream>
#include "core/utils/TypesUtils.h"
#include "core/pack/UnivPack.h"

namespace keops {


// A "zero" vector of size _DIM
// Declared using the   Zero<DIM>   syntax.
template < int _DIM >
struct Zero {
  static const int DIM = _DIM;

  static void PrintId(::std::stringstream& str) {
    str << "0(DIM=" << DIM << ")";
  }

  template<class A, class B>
  using Replace = Zero<DIM>;

  template<class A1, class B1, class A2, class B2>
  using ReplaceVars2 = Zero<DIM>;

  using AllTypes = univpack<Zero<DIM>>;

  template < int CAT >      // Whatever CAT...
  using VARS = univpack<>;  // there's no variable used in there.

  // Evaluation is easy : simply fill-up *out with zeros.
  template < class INDS, typename TYPE, typename... ARGS >
  static DEVICE INLINE void Eval(TYPE* out, ARGS... args) {
    VectAssign<DIM>(out, 0.0f);
  }

  // There is no gradient to accumulate on V, whatever V.
  template < class V, class GRADIN >
  using DiffT = Zero<V::DIM>;
};

#define Zero(D) KeopsNS<Zero<D>>()

}
